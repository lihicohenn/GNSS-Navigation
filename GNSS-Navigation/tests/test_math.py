"""Self-tests for the math and parsing modules.

Run with:  python -m pytest tests/ -q     (or)     python tests/test_math.py

These don't need any external data — they use tiny inline fixtures and known
physical invariants (e.g. a GPS satellite orbits at ~26,560 km and ~3.87 km/s).
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnss.constants import C
from gnss.coords import ecef_to_enu_matrix, ecef_to_geodetic, geodetic_to_ecef
from gnss.ephemeris import GpsEphemeris, sat_pos_vel_clock
from gnss.positioning import earth_rotation_correction, least_squares_position
from gnss.rinex_nav import _parse_gps_record
from gnss.rinex_obs import parse_obs


def test_coords_roundtrip():
    # A point in Israel (near the recording site in the assignment).
    lat, lon, alt = 32.10, 34.85, 45.0
    ecef = geodetic_to_ecef(lat, lon, alt)
    lat2, lon2, alt2 = ecef_to_geodetic(*ecef)
    assert abs(lat - lat2) < 1e-7, (lat, lat2)
    assert abs(lon - lon2) < 1e-7, (lon, lon2)
    assert abs(alt - alt2) < 1e-3, (alt, alt2)
    print("ok  coords round-trip")


def test_ephemeris_orbit_sane():
    # Realistic GPS broadcast values (magnitudes typical of IS-GPS-200).
    eph = GpsEphemeris(
        sat="G01", gps_week=2405, toc=302400.0, toe=302400.0,
        af0=0.0, af1=0.0, af2=0.0,
        sqrt_a=5153.65, e=0.005, m0=0.3, delta_n=4.5e-9,
        omega0=-0.6, i0=0.97, omega=0.8, omega_dot=-8.0e-9, idot=1.0e-10,
        cuc=1e-6, cus=8e-6, crc=250.0, crs=-20.0, cic=-1e-7, cis=1e-7,
        tgd=-1e-8,
    )
    t = 302400.0
    pos, vel, clk, ek = sat_pos_vel_clock(eph, t)
    pos = np.array(pos)
    radius = np.linalg.norm(pos)
    assert 2.5e7 < radius < 2.7e7, radius       # GPS orbital radius ~26,560 km

    # The rigorous velocity check: the analytic ECEF velocity must match a
    # finite-difference of the ECEF position. (ECEF speed differs from the
    # 3.87 km/s *inertial* speed because the ECEF frame co-rotates with Earth.)
    dt = 1e-3
    pos2 = np.array(sat_pos_vel_clock(eph, t + dt)[0])
    numeric_vel = (pos2 - pos) / dt
    assert np.linalg.norm(np.array(vel) - numeric_vel) < 1e-3, (vel, numeric_vel)
    print(f"ok  ephemeris orbit: r={radius/1e3:.0f} km, "
          f"|v_ecef|={np.linalg.norm(vel):.0f} m/s (matches numeric)")


def _fmt_d(v):
    """Format a float in RINEX D19.12 (19-column) form."""
    s = f"{v: .12E}"           # e.g. ' 1.234567890123E+04'
    # Python uses 2-digit exponents already; ensure width 19.
    return s.rjust(19)


def test_nav_record_roundtrip():
    # Build an 8-line GPS record with correct column alignment, then parse it.
    epoch_prefix = "G01 2026 03 21 12 00 00"
    line0 = epoch_prefix + _fmt_d(5e-4) + _fmt_d(1e-11) + _fmt_d(0.0)
    orbit = [
        [10.0, -20.0, 4.5e-9, 0.3],          # IODE, Crs, dn, M0
        [1e-6, 0.005, 8e-6, 5153.65],        # Cuc, e, Cus, sqrtA
        [302400.0, -1e-7, -0.6, 1e-7],       # Toe, Cic, OMEGA0, Cis
        [0.97, 250.0, 0.8, -8e-9],           # i0, Crc, omega, OMEGADOT
        [1e-10, 0.0, 2405.0, 0.0],           # IDOT, codesL2, week, L2P
        [2.0, 0.0, -1e-8, 10.0],             # acc, health, TGD, IODC
        [0.0, 0.0, 0.0, 0.0],                # transmit time, fit...
    ]
    lines = [line0] + ["    " + "".join(_fmt_d(v) for v in row) for row in orbit]
    eph = _parse_gps_record(lines)
    assert eph.sat == "G01"
    assert abs(eph.e - 0.005) < 1e-9
    assert abs(eph.sqrt_a - 5153.65) < 1e-6
    assert abs(eph.toe - 302400.0) < 1e-3
    assert eph.gps_week == 2405
    print("ok  nav record round-trip")


def test_positioning_recovers_known_point():
    """Least-squares must recover a known receiver position from clean ranges."""
    true_pos = geodetic_to_ecef(32.10, 34.85, 45.0)
    true_cdt = 30.0                              # receiver clock bias [m]

    # Put 7 satellites at ~21,000 km along up-looking directions (local ENU).
    enu_to_ecef = ecef_to_enu_matrix(32.10, 34.85).T
    dirs_enu = [
        [0, 0, 1], [0.5, 0, 0.87], [0, 0.5, 0.87], [-0.5, 0, 0.87],
        [0, -0.5, 0.87], [0.3, 0.3, 0.9], [-0.3, 0.3, 0.9],
    ]
    sat_pos, pseudoranges = [], []
    for d in dirs_enu:
        d = np.array(d, float)
        d /= np.linalg.norm(d)
        sp = true_pos + 21e6 * (enu_to_ecef @ d)
        sat_pos.append(sp)
        # Build the pseudorange exactly as the solver interprets it (Sagnac incl.).
        travel = np.linalg.norm(sp - true_pos) / C
        sp_rot = earth_rotation_correction(sp, travel)
        pseudoranges.append(np.linalg.norm(sp_rot - true_pos) + true_cdt)

    sol = least_squares_position(np.array(sat_pos), np.array(pseudoranges))
    assert sol.converged
    assert np.linalg.norm(sol.pos - true_pos) < 1e-3, np.linalg.norm(sol.pos - true_pos)
    assert abs(sol.clock_bias - true_cdt) < 1e-3, sol.clock_bias
    print(f"ok  positioning: recovered point to "
          f"{np.linalg.norm(sol.pos - true_pos)*1e3:.3f} mm, gdop={sol.gdop:.2f}")


SAMPLE_OBS = """\
     4.01           OBSERVATION DATA    M                   RINEX VERSION / TYPE
G    6 C1C D1C S1C C5Q D5Q S5Q                              SYS / # / OBS TYPES
  2026    03    21    15    14   52.0000000     GPS         TIME OF FIRST OBS
                                                            END OF HEADER
> 2026 03 21 15 14 52.4188789  0  2
G08  21137196.39624      1450.05024        25.30024  21134837.03023      1085.15023        18.30023
G10  21423222.68524     -1577.10024        27.30024
> 2026 03 21 15 14 53.4188789  0  1
G08  21136922.38624      1452.55024        28.40024  21134568.71623      1083.75023        22.50023
"""


def test_obs_parser():
    with tempfile.NamedTemporaryFile("w", suffix=".rnx", delete=False) as fh:
        fh.write(SAMPLE_OBS)
        path = fh.name
    header, epochs = parse_obs(path)
    os.unlink(path)

    assert header.obs_types["G"] == ["C1C", "D1C", "S1C", "C5Q", "D5Q", "S5Q"]
    assert len(epochs) == 2
    assert epochs[0].sats["G08"]["C1C"] == 21137196.396
    assert epochs[0].sats["G08"]["D1C"] == 1450.050
    assert epochs[0].sats["G08"]["S1C"] == 25.300
    assert epochs[0].sats["G08"]["C5Q"] == 21134837.030
    assert "C1C" not in epochs[0].sats["G10"] or epochs[0].sats["G10"].get("C5Q") is None
    assert epochs[1].sats["G08"]["C1C"] == 21136922.386
    print("ok  obs parser")


if __name__ == "__main__":
    test_coords_roundtrip()
    test_ephemeris_orbit_sane()
    test_nav_record_roundtrip()
    test_positioning_recovers_known_point()
    test_obs_parser()
    print("\nAll self-tests passed.")
