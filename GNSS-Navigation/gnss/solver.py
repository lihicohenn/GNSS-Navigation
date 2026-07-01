"""Per-epoch orchestration: RINEX measurements + ephemeris -> a full fix.

This module glues the pieces together for one 1 Hz epoch:

    1. choose usable satellites (valid pseudorange + available ephemeris)
    2. compute each satellite's transmit time and propagate its orbit
    3. correct the pseudorange for the satellite clock (+ group delay)
    4. least-squares solve position & clock bias
    5. reject gross outliers and re-solve
    6. least-squares solve velocity from Doppler
    7. convert to geodetic + UTC and package an :class:`EpochSolution`
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

from . import timeutils
from .constants import C, FREQ_L1, SYS_GPS
from .coords import ecef_to_enu_matrix, ecef_to_geodetic
from .ephemeris import GpsEphemeris, sat_pos_vel_clock, select_ephemeris
from .positioning import (
    least_squares_position,
    least_squares_velocity,
    weight_from_cn0,
)
from .rinex_obs import Epoch

LAMBDA_L1 = C / FREQ_L1

# Preferred RINEX observation codes for GPS L1, in priority order.
_PSEUDORANGE_CODES = ("C1C", "C1X", "C1P", "C1W")
_DOPPLER_CODES = ("D1C", "D1X", "D1P", "D1W")
_CN0_CODES = ("S1C", "S1X", "S1P", "S1W")


def _pick(values: dict[str, float], codes) -> float | None:
    for c in codes:
        if c in values:
            return values[c]
    return None


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


@dataclass
class _SatMeasurement:
    sat: str
    sat_pos: np.ndarray
    sat_vel: np.ndarray
    pr_corrected: float
    range_rate: float | None
    sat_clock_rate: float
    cn0: float


def _prepare_satellite(
    sat: str, values: dict, ephs: list[GpsEphemeris], tow: float
) -> _SatMeasurement | None:
    """Compute satellite state and corrected pseudorange for one satellite."""
    pr = _pick(values, _PSEUDORANGE_CODES)
    if pr is None:
        return None
    eph = select_ephemeris(ephs, tow - pr / C)
    if eph is None:
        return None

    # Transmit time, refined once for the satellite clock offset.
    t_tx = tow - pr / C
    _, _, dts, _ = sat_pos_vel_clock(eph, t_tx)
    t_tx = tow - pr / C - dts
    pos, vel, dts, _ = sat_pos_vel_clock(eph, t_tx)

    dts_eff = dts - eph.tgd                          # single-frequency L1 group delay
    pr_corrected = pr + C * dts_eff

    doppler = _pick(values, _DOPPLER_CODES)
    range_rate = None if doppler is None else -LAMBDA_L1 * doppler

    # satellite clock drift (m/s): derivative of the clock polynomial
    from .ephemeris import _time_from_toe
    tk = _time_from_toe(t_tx, eph.toc)
    sat_clock_rate = C * (eph.af1 + 2.0 * eph.af2 * tk)

    cn0 = _pick(values, _CN0_CODES) or 30.0
    return _SatMeasurement(
        sat=sat,
        sat_pos=np.array(pos),
        sat_vel=np.array(vel),
        pr_corrected=pr_corrected,
        range_rate=range_rate,
        sat_clock_rate=sat_clock_rate,
        cn0=cn0,
    )


def solve_epoch(
    epoch: Epoch,
    nav: dict[str, list[GpsEphemeris]],
    prev_pos: np.ndarray | None = None,
    min_sats: int = 4,
    outlier_threshold_m: float = 150.0,
) -> EpochSolution | None:
    """Compute a single 1 Hz fix, or return None if the epoch is unusable."""
    _, tow = timeutils.datetime_to_gps_week_tow(epoch.time)

    measurements: list[_SatMeasurement] = []
    for sat, values in epoch.sats.items():
        if not sat.startswith(SYS_GPS):             # GPS-only milestone
            continue
        m = _prepare_satellite(sat, values, nav.get(sat, []), tow)
        if m is not None:
            measurements.append(m)

    if len(measurements) < min_sats:
        return None

    def solve(ms: list[_SatMeasurement]):
        sat_pos = np.array([m.sat_pos for m in ms])
        pr = np.array([m.pr_corrected for m in ms])
        w = np.array([weight_from_cn0(m.cn0) for m in ms])
        return least_squares_position(sat_pos, pr, weights=w, x0=prev_pos)

    sol = solve(measurements)

    # one round of gross-outlier rejection (drop the single worst if it's bad)
    if len(measurements) > min_sats:
        worst = int(np.argmax(np.abs(sol.residuals)))
        if abs(sol.residuals[worst]) > outlier_threshold_m:
            measurements.pop(worst)
            sol = solve(measurements)

    if not sol.converged:
        return None

    # ---- velocity from Doppler (only sats that actually reported Doppler) ----
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
            weights=np.array([weight_from_cn0(m.cn0) for m in vel_ms]),
        )

    lat, lon, alt = ecef_to_geodetic(*sol.pos)
    vel_enu = ecef_to_enu_matrix(lat, lon) @ vel_ecef

    return EpochSolution(
        time_gps=epoch.time,
        time_utc=timeutils.gps_to_utc(epoch.time),
        lat=lat, lon=lon, alt=alt,
        ecef=sol.pos,
        vel_ecef=vel_ecef,
        vel_enu=vel_enu,
        speed=float(np.linalg.norm(vel_ecef)),
        clock_bias_m=sol.clock_bias,
        clock_drift_ms=clock_drift,
        n_sats=len(measurements),
        gdop=sol.gdop,
    )
