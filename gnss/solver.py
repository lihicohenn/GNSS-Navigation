"""Per-epoch orchestration: RINEX measurements + ephemeris -> a full fix.

This module glues the pieces together for one 1 Hz epoch:

    1. choose usable satellites (valid pseudorange + available ephemeris), across
       every constellation present (GPS / Galileo / BeiDou / GLONASS / QZSS)
    2. compute each satellite's transmit time and propagate its orbit
    3. correct the pseudorange for the satellite clock (+ group delay)
    4. least-squares solve position, receiver clock and inter-system biases
    5. apply ionospheric (Klobuchar) + tropospheric (Saastamoinen) corrections,
       mask low-elevation satellites, and re-solve
    6. reject a gross outlier and re-solve
    7. least-squares solve velocity from Doppler
    8. convert to geodetic + UTC and package an :class:`EpochSolution`
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

from . import timeutils
from .atmosphere import (
    azimuth_elevation,
    klobuchar_delay,
    saastamoinen_delay,
)
from .constants import (
    C,
    FREQ_L1,
    SYS_BEIDOU,
    SYS_GALILEO,
    SYS_GLONASS,
    SYS_GPS,
    SYS_QZSS,
    carrier_frequency,
)
from .coords import ecef_to_enu_matrix, ecef_to_geodetic
from .ephemeris import (
    GlonassEphemeris,
    _time_from_toe,
    gps_tow_to_sys,
    select_ephemeris,
    state_pos_vel_clock,
)
from .positioning import (
    least_squares_position,
    least_squares_velocity,
    weight_from_cn0,
)
from .rinex_nav import NavData
from .rinex_obs import Epoch

# Constellations we solve for, in reference-clock priority order (index 0 is the
# reference receiver clock; the rest are estimated relative to it as ISBs).
SYSTEM_PRIORITY = (SYS_GPS, SYS_GALILEO, SYS_BEIDOU, SYS_GLONASS, SYS_QZSS)

# Pseudorange observation codes per system, in preference order.  The Doppler
# and C/N0 codes are derived from the chosen pseudorange code's band + tracking
# attribute (e.g. 'C1C' -> 'D1C' / 'S1C').
_PR_PREFERENCE = {
    SYS_GPS: ("C1C", "C1W", "C1X", "C1L", "C1P", "C2W", "C2X", "C5Q", "C5X"),
    SYS_GALILEO: ("C1C", "C1X", "C1B", "C1Z", "C5Q", "C5X", "C7Q", "C7X"),
    SYS_BEIDOU: ("C2I", "C2Q", "C2X", "C1P", "C1X", "C1D", "C6I", "C7I", "C5X"),
    SYS_QZSS: ("C1C", "C1X", "C1L", "C2X", "C5Q", "C5X"),
    SYS_GLONASS: ("C1C", "C1P", "C2C", "C2P", "C3Q", "C3X"),
}


def _pick_code(values: dict, prefix: str, band: str, attr: str) -> float | None:
    """Return the value for prefix+band+attr, falling back to same-band, then any."""
    exact = prefix + band + attr
    if exact in values:
        return values[exact]
    for code, val in values.items():          # same measurement type + band
        if code[0] == prefix and code[1] == band:
            return val
    for code, val in values.items():          # same measurement type, any band
        if code[0] == prefix:
            return val
    return None


def _select_observations(values: dict, system: str):
    """Pick (pr, doppler, cn0, band) for a satellite from its RINEX observations."""
    pr_code = None
    for code in _PR_PREFERENCE.get(system, ()):
        if code in values:
            pr_code = code
            break
    if pr_code is None:                       # any pseudorange as a last resort
        candidates = sorted(c for c in values if c.startswith("C"))
        if not candidates:
            return None
        pr_code = candidates[0]

    band, attr = pr_code[1], pr_code[2:]
    pr = values[pr_code]
    doppler = _pick_code(values, "D", band, attr)
    cn0 = _pick_code(values, "S", band, attr)
    return pr, doppler, cn0, band


@dataclass
class SatInfo:
    """Per-satellite diagnostics for one epoch (used by spoofing analysis)."""

    sat: str
    system: str
    cn0: float
    elevation_deg: float
    azimuth_deg: float
    residual_m: float
    iono_m: float
    tropo_m: float
    used: bool


@dataclass
class EpochSolution:
    time_gps: dt.datetime
    time_utc: dt.datetime
    lat: float
    lon: float
    alt: float
    ecef: np.ndarray
    vel_ecef: np.ndarray
    vel_enu: np.ndarray
    speed: float
    clock_bias_m: float
    clock_drift_ms: float
    n_sats: int
    gdop: float
    systems: str = ""                                    # e.g. 'CEG'
    isb_m: dict[str, float] = field(default_factory=dict)  # inter-system biases
    sats: list[SatInfo] = field(default_factory=list)      # per-satellite detail


@dataclass
class _SatMeasurement:
    sat: str
    system: str
    sat_pos: np.ndarray
    sat_vel: np.ndarray
    pr_corrected: float          # satellite-clock-corrected pseudorange [m]
    range_rate: float | None
    sat_clock_rate: float
    cn0: float
    freq_hz: float
    # filled in once an approximate position is known:
    iono: float = 0.0
    tropo: float = 0.0
    elev_deg: float = 0.0
    az_deg: float = 0.0
    residual: float = 0.0

    @property
    def pr_solve(self) -> float:
        return self.pr_corrected - self.iono - self.tropo


def _prepare_satellite(sat: str, values: dict, ephs: list, tow: float) -> _SatMeasurement | None:
    """Compute satellite state and corrected pseudorange for one satellite."""
    system = sat[0]
    obs = _select_observations(values, system)
    if obs is None:
        return None
    pr, doppler, cn0, band = obs

    # Reject non-physical pseudoranges (GnssLogger sometimes emits garbage for a
    # satellite it cannot fully resolve): valid GNSS ranges are ~19,000-42,000 km.
    if not 1.5e7 < pr < 5.0e7:
        return None

    # transmit time, refined once for the satellite clock offset (system scale)
    sys_tow = gps_tow_to_sys(system, tow - pr / C)
    eph = select_ephemeris(ephs, sys_tow)
    if eph is None:
        return None
    _, _, dts, _ = state_pos_vel_clock(eph, sys_tow)
    sys_tow = gps_tow_to_sys(system, tow - pr / C - dts)
    pos, vel, dts, _ = state_pos_vel_clock(eph, sys_tow)

    tgd = getattr(eph, "tgd", 0.0)                       # GLONASS has no broadcast TGD
    pr_corrected = pr + C * (dts - tgd)

    freq_num = getattr(eph, "freq_num", 0)               # GLONASS FDMA channel
    freq_hz = carrier_frequency(system, band, freq_num)

    range_rate = None if doppler is None else -(C / freq_hz) * doppler

    # satellite clock drift [m/s]
    if isinstance(eph, GlonassEphemeris):
        sat_clock_rate = C * eph.gamma_n
    else:
        tk = _time_from_toe(sys_tow, eph.toc)
        sat_clock_rate = C * (eph.af1 + 2.0 * eph.af2 * tk)

    return _SatMeasurement(
        sat=sat, system=system,
        sat_pos=np.array(pos), sat_vel=np.array(vel),
        pr_corrected=pr_corrected,
        range_rate=range_rate,
        sat_clock_rate=sat_clock_rate,
        cn0=cn0 if cn0 is not None else 30.0,
        freq_hz=freq_hz,
    )


def _system_indices(measurements: list[_SatMeasurement]) -> tuple[list[str], dict[str, int]]:
    """Order present systems by priority and map each to a clock-column index."""
    present = {m.system for m in measurements}
    ordered = [s for s in SYSTEM_PRIORITY if s in present]
    ordered += sorted(present - set(ordered))            # any unexpected system
    return ordered, {s: i for i, s in enumerate(ordered)}


def _weights(measurements: list[_SatMeasurement], use_elev: bool) -> np.ndarray:
    """C/N0 weighting, optionally de-weighting low-elevation satellites."""
    w = np.array([weight_from_cn0(m.cn0) for m in measurements])
    if use_elev:
        el = np.array([max(np.radians(m.elev_deg), np.radians(5.0)) for m in measurements])
        w = w * np.sin(el) ** 2
    return w


def solve_epoch(
    epoch: Epoch,
    nav: NavData,
    prev_pos: np.ndarray | None = None,
    min_sats: int = 4,
    outlier_threshold_m: float = 150.0,
    corrections: bool = True,
    elevation_mask_deg: float = 5.0,
    systems: str | None = None,
    max_gdop: float = 30.0,
    alt_bounds_m: tuple[float, float] = (-1000.0, 9000.0),
) -> EpochSolution | None:
    """Compute a single 1 Hz fix, or return None if the epoch is unusable.

    ``systems`` optionally restricts the constellations used (e.g. ``"G"`` for a
    GPS-only fix, ``"GE"`` for GPS+Galileo).  ``corrections`` toggles the
    ionospheric/tropospheric models and the elevation mask.  A fix is rejected
    (returns None) if its geometry is too weak (GDOP > ``max_gdop``) or its
    altitude is non-physical for a hand-held recording — these plausibility gates
    stop a bad epoch from emitting a wild position and poisoning the next epoch's
    seed.
    """
    _, tow = timeutils.datetime_to_gps_week_tow(epoch.time)

    measurements: list[_SatMeasurement] = []
    for sat, values in epoch.sats.items():
        if sat[0] not in SYSTEM_PRIORITY:
            continue
        if systems is not None and sat[0] not in systems:
            continue
        m = _prepare_satellite(sat, values, nav.eph.get(sat, []), tow)
        if m is not None:
            measurements.append(m)

    def run_solve(ms: list[_SatMeasurement], use_elev: bool):
        ordered, idx = _system_indices(ms)
        sys_index = np.array([idx[m.system] for m in ms])
        sol = least_squares_position(
            sat_positions=np.array([m.sat_pos for m in ms]),
            pseudoranges=np.array([m.pr_solve for m in ms]),
            weights=_weights(ms, use_elev),
            x0=prev_pos,
            sys_index=sys_index,
        )
        return sol, ordered, idx

    def enough(ms: list[_SatMeasurement]) -> bool:
        n_unknown = 3 + len({m.system for m in ms})
        return len(ms) >= max(min_sats, n_unknown)

    if not enough(measurements):
        return None

    # ---- first (uncorrected) solve to get an approximate position ----
    sol, ordered, idx = run_solve(measurements, use_elev=False)
    if not sol.converged:
        return None

    # ---- atmospheric corrections + elevation mask, then re-solve ----
    if corrections:
        lat0, _, height0 = ecef_to_geodetic(*sol.pos)
        kept = []
        for m in measurements:
            az, el = azimuth_elevation(sol.pos, m.sat_pos)
            m.az_deg, m.elev_deg = np.degrees(az), np.degrees(el)
            if nav.iono_alpha is not None and nav.iono_beta is not None:
                iono_l1 = klobuchar_delay(
                    sol.pos, az, el, tow, nav.iono_alpha, nav.iono_beta
                )
                m.iono = iono_l1 * (FREQ_L1 / m.freq_hz) ** 2
            m.tropo = saastamoinen_delay(el, lat0, height0)
            if m.elev_deg >= elevation_mask_deg:
                kept.append(m)
        if enough(kept):
            measurements = kept
        if enough(measurements):
            sol, ordered, idx = run_solve(measurements, use_elev=True)
            if not sol.converged:
                return None

    # ---- iterative gross-outlier rejection (RAIM-style) ----
    # Smartphone measurements frequently carry several multipath/NLOS outliers,
    # so drop the worst residual and re-solve, repeating while any residual
    # exceeds the threshold and enough satellites remain for a redundant fix.
    while len(measurements) > 3 + len(ordered):
        worst = int(np.argmax(np.abs(sol.residuals)))
        if abs(sol.residuals[worst]) <= outlier_threshold_m:
            break
        measurements.pop(worst)
        if not enough(measurements):
            break
        sol, ordered, idx = run_solve(measurements, use_elev=corrections)
        if not sol.converged:
            break

    if not sol.converged:
        return None

    # ---- plausibility gates: reject a geometrically weak or non-physical fix ----
    if sol.gdop > max_gdop:
        return None
    lat, lon, alt = ecef_to_geodetic(*sol.pos)
    if not alt_bounds_m[0] <= alt <= alt_bounds_m[1]:
        return None

    for m, r in zip(measurements, sol.residuals):
        m.residual = float(r)

    # ---- velocity from Doppler (single receiver clock drift, all systems) ----
    vel_ecef = np.zeros(3)
    clock_drift = 0.0
    vel_ms = [m for m in measurements if m.range_rate is not None]
    if len(vel_ms) >= 4:
        vel_ecef, clock_drift = least_squares_velocity(
            sat_positions=np.array([m.sat_pos for m in vel_ms]),
            sat_velocities=np.array([m.sat_vel for m in vel_ms]),
            rcv_pos=sol.pos,
            range_rates=np.array([m.range_rate for m in vel_ms]),
            sat_clock_rates=np.array([m.sat_clock_rate for m in vel_ms]),
            weights=_weights(vel_ms, use_elev=corrections),
        )

    vel_enu = ecef_to_enu_matrix(lat, lon) @ vel_ecef

    ref_bias = sol.clock_biases[0]
    isb = {s: float(sol.clock_biases[idx[s]] - ref_bias) for s in ordered}
    sat_infos = [
        SatInfo(
            sat=m.sat, system=m.system, cn0=m.cn0,
            elevation_deg=m.elev_deg, azimuth_deg=m.az_deg,
            residual_m=m.residual, iono_m=m.iono, tropo_m=m.tropo, used=True,
        )
        for m in measurements
    ]

    return EpochSolution(
        time_gps=epoch.time,
        time_utc=timeutils.gps_to_utc(epoch.time),
        lat=lat, lon=lon, alt=alt,
        ecef=sol.pos,
        vel_ecef=vel_ecef,
        vel_enu=vel_enu,
        speed=float(np.linalg.norm(vel_ecef)),
        clock_bias_m=ref_bias,
        clock_drift_ms=clock_drift,
        n_sats=len(measurements),
        gdop=sol.gdop,
        systems="".join(ordered),
        isb_m=isb,
        sats=sat_infos,
    )
