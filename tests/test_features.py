"""Self-tests for the multi-constellation, atmospheric, NMEA and spoofing work.

Like the other test modules, these need no external data: they forward-model
self-consistent measurements (a true fixed point of the solver) and check that
the pipeline recovers the truth, or exercise a model against known physical
bounds.

Run:  python tests/test_features.py
"""

import datetime as dt
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnss.atmosphere import (
    azimuth_elevation,
    klobuchar_delay,
    saastamoinen_delay,
)
from gnss.constants import C, FREQ_L1, FREQ_L5
from gnss.coords import ecef_to_enu_matrix, ecef_to_geodetic, geodetic_to_ecef
from gnss.ephemeris import (
    GlonassEphemeris,
    GpsEphemeris,
    gps_tow_to_sys,
    sat_pos_vel_clock,
    state_pos_vel_clock,
)
from gnss.nmea import parse_nmea
from gnss.positioning import earth_rotation_correction, least_squares_position
from gnss.rinex_obs import Epoch
from gnss.rinex_nav import NavData
from gnss.solver import EpochSolution, SatInfo, solve_epoch
from gnss.timeutils import GPS_EPOCH
from gnss import spoofing, validate

SITE = (32.10, 34.85, 45.0)
WEEK, TOW0 = 2405, 302400.0
# calendar time (GPS scale) that corresponds exactly to (WEEK, TOW0) — so the
# solver recovers the same seconds-of-week the forward model propagates with.
EPOCH_TIME = GPS_EPOCH + dt.timedelta(seconds=WEEK * 604800 + TOW0)

# A realistic broadcast Klobuchar set (mid-latitude, moderate activity).
IONO_ALPHA = (1.1176e-08, -7.4506e-09, -5.9605e-08, 1.1921e-07)
IONO_BETA = (1.1674e05, -2.2938e05, -1.3107e05, 1.0486e06)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _kepler_eph(prn, sqrt_a, m0, omega0, i0=0.96, e=0.004):
    return GpsEphemeris(
        sat=prn, gps_week=WEEK, toc=TOW0, toe=TOW0,
        af0=0.0, af1=0.0, af2=0.0,
        sqrt_a=sqrt_a, e=e, m0=m0, delta_n=4.5e-9,
        omega0=omega0, i0=i0, omega=0.7, omega_dot=-8.0e-9, idot=1.0e-10,
        cuc=1e-6, cus=8e-6, crc=250.0, crs=-20.0, cic=-1e-7, cis=1e-7,
        tgd=-5e-9,
    )


def _glonass_eph(prn, pos, vel):
    return GlonassEphemeris(
        sat=prn, toe=TOW0, toc=TOW0, tau_n=-1e-4, gamma_n=1e-12,
        pos=np.array(pos, float), vel=np.array(vel, float),
        acc=np.array([1e-6, -1e-6, 2e-6]) * 1e3, freq_num=1,
    )


def _visible_kepler_ephs(prefix, sqrt_a, rcv, n, start_prn=1, i0=0.96):
    """Grid-search Keplerian elements for `n` satellites above 30 deg at `rcv`."""
    found, prn = [], start_prn
    for m0 in np.linspace(0.0, 2 * math.pi, 16, endpoint=False):
        for omega0 in np.linspace(-math.pi, math.pi, 16, endpoint=False):
            eph = _kepler_eph(f"{prefix}{prn:02d}", sqrt_a, m0, omega0, i0=i0)
            pos = np.array(sat_pos_vel_clock(eph, gps_tow_to_sys(prefix, TOW0))[0])
            _, el = azimuth_elevation(rcv, pos)
            if math.degrees(el) > 30.0:
                found.append(eph)
                prn += 1
                if len(found) >= n:
                    return found
    return found


def _place_glonass(prn, rcv, az_deg, el_deg, radius=25.51e6):
    """Build a GLONASS ephemeris whose satellite sits at a given az/el, radius."""
    lat, lon, _ = ecef_to_geodetic(*rcv)
    enu_to_ecef = ecef_to_enu_matrix(lat, lon).T
    az, el = math.radians(az_deg), math.radians(el_deg)
    unit = np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az),
                     math.sin(el)])
    d = enu_to_ecef @ unit
    b = float(rcv @ d)
    s = -b + math.sqrt(b * b - (float(rcv @ rcv) - radius**2))
    pos = rcv + s * d
    vperp = np.cross(pos, [0.0, 0.0, 1.0])
    vel = vperp / np.linalg.norm(vperp) * 3900.0
    return _glonass_eph(prn, pos, vel)


def _consistent_pr(eph, tow, rcv, cdt_sys):
    """Pseudorange that is an exact fixed point of the solver (given ISB cdt_sys)."""
    system = eph.system
    pr = 22_000_000.0
    pos = vel = None
    dts = 0.0
    for _ in range(12):
        sys_tow = gps_tow_to_sys(system, tow - pr / C)
        _, _, dts, _ = state_pos_vel_clock(eph, sys_tow)
        sys_tow = gps_tow_to_sys(system, tow - pr / C - dts)
        pos, vel, dts, _ = state_pos_vel_clock(eph, sys_tow)
        pos = np.array(pos)
        travel = np.linalg.norm(pos - rcv) / C
        pos_rot = earth_rotation_correction(pos, travel)
        rng = np.linalg.norm(pos_rot - rcv)
        tgd = getattr(eph, "tgd", 0.0)
        pr = rng + cdt_sys - C * (dts - tgd)
    return pr, np.array(pos), np.array(vel)


# ---------------------------------------------------------------------------
# GLONASS orbit integration
# ---------------------------------------------------------------------------
def test_glonass_orbit():
    # A self-consistent-ish PZ-90 state at GLONASS altitude (~25,510 km radius).
    eph = _glonass_eph("R07", [7003.0e3, -12206.0e3, 21280.0e3],
                       [0.7e3, -2.85e3, -1.9e3])
    pos, vel, clk, _ = state_pos_vel_clock(eph, TOW0)     # at toe -> input state
    r = np.linalg.norm(pos)
    assert 2.5e7 < r < 2.6e7, r

    # analytic (integrated) velocity must match a finite difference
    dt = 1e-3
    pos2 = np.array(state_pos_vel_clock(eph, TOW0 + dt)[0])
    numeric_v = (pos2 - np.array(pos)) / dt
    assert np.linalg.norm(np.array(vel) - numeric_v) < 1e-3, (vel, numeric_v)
    print(f"ok  GLONASS orbit: r={r/1e3:.0f} km, velocity matches numeric diff")


# ---------------------------------------------------------------------------
# BeiDou GEO vs MEO branches
# ---------------------------------------------------------------------------
def test_beidou_geo_and_meo():
    geo = _kepler_eph("C01", sqrt_a=6493.4, m0=0.1, omega0=1.0, i0=0.09)
    meo = _kepler_eph("C21", sqrt_a=5282.6, m0=0.3, omega0=-1.0, i0=0.96)
    pos_geo = np.array(sat_pos_vel_clock(geo, gps_tow_to_sys("C", TOW0))[0])
    pos_meo = np.array(sat_pos_vel_clock(meo, gps_tow_to_sys("C", TOW0))[0])
    r_geo, r_meo = np.linalg.norm(pos_geo), np.linalg.norm(pos_meo)
    assert 4.1e7 < r_geo < 4.3e7, r_geo        # geostationary radius ~42,164 km
    assert 2.6e7 < r_meo < 2.9e7, r_meo        # BeiDou MEO radius ~27,900 km

    # MEO analytic velocity matches finite difference
    v_meo = np.array(sat_pos_vel_clock(meo, gps_tow_to_sys("C", TOW0))[1])
    dt = 1e-3
    pos2 = np.array(sat_pos_vel_clock(meo, gps_tow_to_sys("C", TOW0 + dt))[0])
    assert np.linalg.norm(v_meo - (pos2 - pos_meo) / dt) < 1e-3
    print(f"ok  BeiDou GEO r={r_geo/1e3:.0f} km, MEO r={r_meo/1e3:.0f} km")


# ---------------------------------------------------------------------------
# inter-system clock bias recovery (least-squares linear algebra)
# ---------------------------------------------------------------------------
def test_inter_system_bias():
    true_pos = geodetic_to_ecef(*SITE)
    cdt = {"G": 20.0, "E": 25.0, "C": 12.0}    # absolute per-system clocks [m]
    enu_to_ecef = ecef_to_enu_matrix(SITE[0], SITE[1]).T
    dirs = [[0, 0, 1], [0.6, 0, 0.8], [0, 0.6, 0.8], [-0.6, 0, 0.8],
            [0, -0.6, 0.8], [0.4, 0.4, 0.82], [-0.4, 0.4, 0.82], [0.4, -0.4, 0.82],
            [-0.4, -0.4, 0.82]]
    systems = ["G", "G", "G", "E", "E", "E", "C", "C", "C"]

    sat_pos, prs, sys_index = [], [], []
    sys_to_idx = {"G": 0, "E": 1, "C": 2}
    for d, s in zip(dirs, systems):
        d = np.array(d, float)
        d /= np.linalg.norm(d)
        sp = true_pos + 21e6 * (enu_to_ecef @ d)
        travel = np.linalg.norm(sp - true_pos) / C
        sp_rot = earth_rotation_correction(sp, travel)
        sat_pos.append(sp)
        prs.append(np.linalg.norm(sp_rot - true_pos) + cdt[s])
        sys_index.append(sys_to_idx[s])

    sol = least_squares_position(np.array(sat_pos), np.array(prs),
                                 sys_index=np.array(sys_index))
    assert sol.converged
    assert np.linalg.norm(sol.pos - true_pos) < 1e-3, np.linalg.norm(sol.pos - true_pos)
    for s, i in sys_to_idx.items():
        assert abs(sol.clock_biases[i] - cdt[s]) < 1e-3, (s, sol.clock_biases[i])
    print(f"ok  inter-system bias: pos to "
          f"{np.linalg.norm(sol.pos - true_pos)*1e3:.3f} mm, "
          f"ISB(E-G)={sol.clock_biases[1]-sol.clock_biases[0]:.2f} m, "
          f"ISB(C-G)={sol.clock_biases[2]-sol.clock_biases[0]:.2f} m")


# ---------------------------------------------------------------------------
# full multi-constellation solve_epoch end to end (GPS+GAL+BDS+GLO)
# ---------------------------------------------------------------------------
def _build_multignss_epoch(rcv, cdt):
    ephs = {}
    for e in _visible_kepler_ephs("G", 5153.65, rcv, 3, start_prn=1):
        ephs[e.sat] = e
    for e in _visible_kepler_ephs("E", 5440.6, rcv, 3, start_prn=11):
        ephs[e.sat] = e
    for e in _visible_kepler_ephs("C", 5282.6, rcv, 3, start_prn=20):  # >5 -> MEO
        ephs[e.sat] = e
    for prn, (az, el) in zip(("R07", "R08", "R09"),
                             ((40, 55), (160, 45), (280, 65))):
        ephs[prn] = _place_glonass(prn, rcv, az, el)
    nav = NavData(eph={k: [v] for k, v in ephs.items()})

    sats = {}
    for prn, eph in ephs.items():
        pr, _, _ = _consistent_pr(eph, TOW0, rcv, cdt[prn[0]])
        pr_code = "C2I" if prn[0] == "C" else "C1C"
        cn0_code = "S2I" if prn[0] == "C" else "S1C"
        sats[prn] = {pr_code: pr, cn0_code: 44.0}

    return Epoch(time=EPOCH_TIME, flag=0, sats=sats), nav


def test_multignss_end_to_end():
    rcv = geodetic_to_ecef(*SITE)
    cdt = {"G": 30.0, "E": 35.0, "C": 18.0, "R": 42.0}    # absolute per-system [m]
    epoch, nav = _build_multignss_epoch(rcv, cdt)

    sol = solve_epoch(epoch, nav, corrections=False, min_sats=4)
    assert sol is not None
    err = np.linalg.norm(sol.ecef - rcv)
    assert err < 1e-2, f"position error {err:.4f} m"
    assert sol.systems == "GECR", sol.systems
    assert abs(sol.clock_bias_m - cdt["G"]) < 1e-2
    for s in "ECR":
        expect = cdt[s] - cdt["G"]
        assert abs(sol.isb_m[s] - expect) < 1e-2, (s, sol.isb_m[s], expect)
    print(f"ok  multi-GNSS end-to-end: 12 sats [{sol.systems}], "
          f"pos err={err*1e3:.3f} mm, ISB {{"
          + ", ".join(f'{s}:{sol.isb_m[s]:.1f}' for s in 'ECR') + "}")


# ---------------------------------------------------------------------------
# atmospheric models
# ---------------------------------------------------------------------------
def test_saastamoinen_troposphere():
    zenith = saastamoinen_delay(math.radians(90.0), SITE[0], 0.0)
    assert 2.2 < zenith < 2.6, zenith                     # ~2.3 m zenith at sea level
    low = saastamoinen_delay(math.radians(30.0), SITE[0], 0.0)
    assert abs(low - zenith / math.sin(math.radians(30.0))) < 0.05
    high = saastamoinen_delay(math.radians(90.0), SITE[0], 2000.0)
    assert high < zenith                                  # thinner atmosphere up high
    assert saastamoinen_delay(math.radians(-1.0), SITE[0], 0.0) == 0.0
    print(f"ok  troposphere: zenith={zenith:.2f} m, 30deg={low:.2f} m, "
          f"2 km={high:.2f} m")


def test_klobuchar_ionosphere():
    rcv = geodetic_to_ecef(*SITE)
    enu_to_ecef = ecef_to_enu_matrix(SITE[0], SITE[1]).T
    zenith_sat = rcv + 21e6 * (enu_to_ecef @ np.array([0, 0, 1.0]))
    low_sat = rcv + 21e6 * (enu_to_ecef @ (np.array([1, 0, 0.2]) /
                                           np.linalg.norm([1, 0, 0.2])))
    tow = 50400.0                                         # local noon-ish -> max iono
    az_z, el_z = azimuth_elevation(rcv, zenith_sat)
    az_l, el_l = azimuth_elevation(rcv, low_sat)

    d_zenith = klobuchar_delay(rcv, az_z, el_z, tow, IONO_ALPHA, IONO_BETA)
    d_low = klobuchar_delay(rcv, az_l, el_l, tow, IONO_ALPHA, IONO_BETA)
    assert 0.5 < d_zenith < 20.0, d_zenith               # metres on L1
    assert d_low > d_zenith                               # obliquity inflates low sats

    # dispersive: L5 delay is larger than L1 by (fL1/fL5)^2 ~ 1.79
    scale = (FREQ_L1 / FREQ_L5) ** 2
    assert abs(scale - 1.79) < 0.05
    print(f"ok  ionosphere: zenith={d_zenith:.2f} m, low-elev={d_low:.2f} m, "
          f"L5/L1 scale={scale:.2f}")


def test_corrections_recovered_end_to_end():
    """Ranges built WITH iono+tropo must be recovered by turning the models on."""
    rcv = geodetic_to_ecef(*SITE)
    lat, _, height = ecef_to_geodetic(*rcv)
    ephs = {e.sat: e for e in _visible_kepler_ephs("G", 5153.65, rcv, 6)}
    nav = NavData(eph={k: [v] for k, v in ephs.items()},
                  iono_alpha=IONO_ALPHA, iono_beta=IONO_BETA)

    sats = {}
    for prn, eph in ephs.items():
        pr, spos, _ = _consistent_pr(eph, TOW0, rcv, cdt_sys=25.0)
        az, el = azimuth_elevation(rcv, spos)
        iono = klobuchar_delay(rcv, az, el, TOW0, IONO_ALPHA, IONO_BETA)
        tropo = saastamoinen_delay(el, lat, height)
        sats[prn] = {"C1C": pr + iono + tropo, "S1C": 45.0}

    epoch = Epoch(time=EPOCH_TIME, flag=0, sats=sats)
    assert len(sats) >= 5, "need enough high-elevation sats for the test"

    off = solve_epoch(epoch, nav, corrections=False, elevation_mask_deg=0.0)
    on = solve_epoch(epoch, nav, corrections=True, elevation_mask_deg=5.0)
    err_off = np.linalg.norm(off.ecef - rcv)
    err_on = np.linalg.norm(on.ecef - rcv)
    assert err_off > 1.0, f"uncorrected error unexpectedly small: {err_off:.2f} m"
    assert err_on < 0.3, f"corrected error too large: {err_on:.2f} m"
    print(f"ok  corrections recovered: error {err_off:.2f} m -> {err_on:.3f} m")


# ---------------------------------------------------------------------------
# NMEA parsing + validation
# ---------------------------------------------------------------------------
_NMEA_BODIES = [
    "GPGGA,151452.00,3206.00000,N,03451.00000,E,1,09,0.9,45.0,M,17.0,M,,",
    "GPRMC,151452.00,A,3206.00000,N,03451.00000,E,1.94,54.7,210326,,,A",
    "GPGGA,151453.00,3206.00540,N,03451.00000,E,1,09,0.9,45.0,M,17.0,M,,",
]


def _nmea_line(body: str) -> str:
    cs = 0
    for ch in body:
        cs ^= ord(ch)
    return f"${body}*{cs:02X}"


def test_nmea_parser():
    path = os.path.join(os.path.dirname(__file__), "_sample.nmea")
    with open(path, "w") as fh:
        fh.write("\n".join(_nmea_line(b) for b in _NMEA_BODIES) + "\n")
    fixes = parse_nmea(path)
    os.unlink(path)

    assert len(fixes) == 2, len(fixes)
    f0 = fixes[0]
    assert abs(f0.lat - 32.10) < 1e-6, f0.lat            # 3206.00000' N = 32.1000 deg
    assert abs(f0.lon - 34.85) < 1e-6, f0.lon            # 03451.00000' E = 34.85 deg
    assert abs(f0.tod - (15 * 3600 + 14 * 60 + 52)) < 1e-6
    assert f0.alt == 45.0 and f0.geoid_sep == 17.0
    assert f0.speed is not None and abs(f0.speed - 1.94 * 0.514444) < 1e-3
    print(f"ok  NMEA parser: {len(fixes)} fixes, lat={f0.lat:.4f}, "
          f"speed={f0.speed:.2f} m/s")


def test_nmea_gnsslogger_wrapper():
    """GnssLogger wraps each sentence as 'NMEA,$GNGGA,...*46,<millis>'."""
    body = "GNGGA,151452.00,3206.00000,N,03451.00000,E,1,12,0.4,45.0,M,17.0,M,,"
    wrapped = "NMEA," + _nmea_line(body) + ",1774106278016"
    path = os.path.join(os.path.dirname(__file__), "_wrap.nmea")
    with open(path, "w") as fh:
        fh.write(wrapped + "\n")
    fixes = parse_nmea(path)
    os.unlink(path)
    assert len(fixes) == 1, len(fixes)
    assert abs(fixes[0].lat - 32.10) < 1e-6 and abs(fixes[0].lon - 34.85) < 1e-6
    print("ok  NMEA GnssLogger wrapper: unwrapped 'NMEA,$..*cs,millis' correctly")


# ---------------------------------------------------------------------------
# regression: RINEX 3.05 GLONASS records carry a 5th line; a fixed 4-line read
# desynchronises and silently drops most GLONASS ephemerides (real-data bug).
# ---------------------------------------------------------------------------
def _fmt_d(v):
    return f"{v: .12E}".rjust(19)


def _glo_record(prn, hh, mm, taun, pos_km, vel_kms):
    epoch = f"{prn} 2026 03 21 {hh:02d} {mm:02d} 00"
    lines = [epoch + _fmt_d(taun) + _fmt_d(0.0) + _fmt_d(0.0)]
    rows = [
        [pos_km[0], vel_kms[0], 0.0, 0.0],
        [pos_km[1], vel_kms[1], 0.0, 1.0],
        [pos_km[2], vel_kms[2], 0.0, 0.0],
        [9.999e08, 15.0, 0.0, 0.0],          # RINEX 3.05 5th (broadcast-orbit 4) line
    ]
    lines += ["    " + "".join(_fmt_d(v) for v in row) for row in rows]
    return lines


def test_glonass_rinex305_5line_records(tmp_path=None):
    import tempfile
    header = [
        "     3.05           NAVIGATION DATA     MIXED               RINEX VERSION / TYPE",
        "GPSA   1.6764e-08  7.4506e-09 -1.1921e-07 -5.9605E-08       IONOSPHERIC CORR",
        "GPSB   1.1059e+05  1.6384e+04 -2.6214e+05 -6.5536E+04       IONOSPHERIC CORR",
        "                                                            END OF HEADER",
    ]
    body = []
    body += _glo_record("R07", 15, 15, -1.5e-4, [7003.0, -12206.0, 21280.0], [0.7, -2.85, -1.9])
    body += _glo_record("R08", 15, 45, -2.0e-4, [-15000.0, 9000.0, 18000.0], [1.2, 1.5, -1.7])
    # a following GPS (8-line) record must still parse -> proves no desync
    body.append("G01 2026 03 21 15 15 00" + _fmt_d(1e-4) + _fmt_d(0.0) + _fmt_d(0.0))
    g = [[0.0, -20.0, 4.5e-9, 0.3], [1e-6, 0.005, 8e-6, 5153.65],
         [TOW0, -1e-7, -0.6, 1e-7], [0.97, 250.0, 0.8, -8e-9],
         [1e-10, 0.0, 2405.0, 0.0], [2.0, 0.0, -1e-8, 10.0], [0.0, 0.0, 0.0, 0.0]]
    body += ["    " + "".join(_fmt_d(v) for v in row) for row in g]

    with tempfile.NamedTemporaryFile("w", suffix=".rnx", delete=False) as fh:
        fh.write("\n".join(header + body) + "\n")
        path = fh.name
    from gnss.rinex_nav import parse_nav
    from gnss.ephemeris import GlonassEphemeris
    nav = parse_nav(path)
    os.unlink(path)

    assert set(nav.eph) == {"R07", "R08", "G01"}, set(nav.eph)   # nothing dropped
    assert isinstance(nav.eph["R07"][0], GlonassEphemeris)
    assert nav.iono_alpha is not None and nav.iono_beta is not None
    r = np.linalg.norm(nav.eph["R08"][0].pos)
    assert 2.4e7 < r < 2.7e7, r                                   # sane GLONASS radius
    print("ok  RINEX 3.05 GLONASS 5-line records: all records parsed, no desync")


def test_pseudorange_sanity_gate():
    """A satellite with a non-physical pseudorange is dropped, not trusted."""
    from gnss.solver import _prepare_satellite
    eph = _kepler_eph("G01", 5153.65, 0.2, -2.4)
    ephs = [eph]
    good = _prepare_satellite("G01", {"C1C": 2.2e7, "S1C": 40.0}, ephs, TOW0)
    bad = _prepare_satellite("G01", {"C1C": 7.5e12, "S1C": 40.0}, ephs, TOW0)
    assert good is not None and bad is None
    print("ok  pseudorange sanity gate: 7.5e12 m dropped, 2.2e7 m kept")


def _fake_solution(index, lat, lon, alt=45.0):
    t = dt.datetime(2026, 3, 21, 15, 14, 52 + index, tzinfo=dt.timezone.utc)
    ecef = geodetic_to_ecef(lat, lon, alt)
    return EpochSolution(
        time_gps=t, time_utc=t, lat=lat, lon=lon, alt=alt, ecef=ecef,
        vel_ecef=np.zeros(3), vel_enu=np.zeros(3), speed=0.0,
        clock_bias_m=0.0, clock_drift_ms=0.0, n_sats=8, gdop=2.0, systems="G",
    )


def test_nmea_validation():
    from gnss.nmea import NmeaFix
    # truth track; our "solution" is offset ~5 m east of it
    lat0, lon0 = 32.10, 34.85
    dlon = 5.0 / (111320.0 * math.cos(math.radians(lat0)))   # ~5 m east in deg lon
    solutions = [_fake_solution(i, lat0, lon0 + dlon) for i in range(3)]
    fixes = [NmeaFix(tod=15 * 3600 + 14 * 60 + 52 + i, lat=lat0, lon=lon0,
                     alt=28.0, geoid_sep=17.0, quality=1) for i in range(3)]

    report = validate.compare(solutions, fixes)
    assert report.n_matched == 3
    assert abs(report.horiz_mean - 5.0) < 0.2, report.horiz_mean
    print(f"ok  NMEA validation: matched {report.n_matched}, "
          f"horiz mean={report.horiz_mean:.2f} m (expected ~5)")


# ---------------------------------------------------------------------------
# spoofing / integrity analysis
# ---------------------------------------------------------------------------
def _solution_with_sats(index, ecef, vel, sat_specs):
    t = dt.datetime(2026, 3, 21, 15, 14, 52 + index, tzinfo=dt.timezone.utc)
    lat, lon, alt = ecef_to_geodetic(*ecef)
    sats = [SatInfo(sat=f"G{j:02d}", system="G", cn0=c, elevation_deg=el,
                    azimuth_deg=0.0, residual_m=r, iono_m=0.0, tropo_m=0.0, used=True)
            for j, (c, el, r) in enumerate(sat_specs)]
    return EpochSolution(
        time_gps=t, time_utc=t, lat=lat, lon=lon, alt=alt, ecef=np.array(ecef),
        vel_ecef=np.array(vel), vel_enu=np.zeros(3), speed=float(np.linalg.norm(vel)),
        clock_bias_m=0.0, clock_drift_ms=0.0, n_sats=len(sats), gdop=2.0,
        systems="G", sats=sats,
    )


def test_spoofing_detection():
    base = geodetic_to_ecef(*SITE)
    # --- clean: C/N0 rises with elevation, small residuals, steady motion ---
    clean_sats = [(38, 20, 1.2), (42, 40, -0.8), (46, 60, 0.5),
                  (48, 75, -1.1), (44, 50, 0.9), (40, 30, -0.6)]
    clean = []
    for i in range(4):
        ecef = base + np.array([i * 1.0, 0.0, 0.0])       # 1 m/s east
        clean.append(_solution_with_sats(i, ecef, [1.0, 0.0, 0.0], clean_sats))
    clean_report = spoofing.analyze(clean)
    assert clean_report.suspicious_epochs == 0, clean_report.flag_counts

    # --- spoofed: uniformly high C/N0, inflated residuals, a position jump ---
    spoof_sats = [(50, 20, 40.0), (50, 40, -38.0), (50, 60, 42.0),
                  (50, 75, -41.0), (50, 50, 39.0), (50, 30, -40.0)]
    spoofed = []
    for i in range(4):
        jump = 500.0 if i == 2 else 0.0                   # sudden 500 m jump
        ecef = base + np.array([i * 1.0 + jump, 0.0, 0.0])
        spoofed.append(_solution_with_sats(i, ecef, [1.0, 0.0, 0.0], spoof_sats))
    spoof_report = spoofing.analyze(spoofed)
    assert spoof_report.suspicious_epochs >= 3, spoof_report.flag_counts
    counts = spoof_report.flag_counts
    assert counts.get("high-residuals", 0) >= 3
    assert counts.get("uniform-power", 0) >= 3
    assert counts.get("kinematic-jump", 0) >= 1
    print(f"ok  spoofing: clean=0 flagged, spoofed="
          f"{spoof_report.suspicious_epochs}/4 flagged {dict(counts)}")


if __name__ == "__main__":
    test_glonass_orbit()
    test_beidou_geo_and_meo()
    test_inter_system_bias()
    test_multignss_end_to_end()
    test_saastamoinen_troposphere()
    test_klobuchar_ionosphere()
    test_corrections_recovered_end_to_end()
    test_nmea_parser()
    test_nmea_gnsslogger_wrapper()
    test_glonass_rinex305_5line_records()
    test_pseudorange_sanity_gate()
    test_nmea_validation()
    test_spoofing_detection()
    print("\nAll feature self-tests passed.")
