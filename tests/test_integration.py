"""End-to-end integration test with synthetic (but self-consistent) RINEX data.

We invent a known moving receiver and a set of GPS satellites, forward-model the
pseudoranges/Doppler exactly the way the solver interprets them, write real
RINEX observation + navigation files, then run the full ``main.py`` pipeline and
check that the recovered CSV trajectory matches the truth to millimetre level.

This validates the whole chain end-to-end: RINEX writing/parsing, ephemeris
propagation, least-squares position + velocity, and CSV output.

Run:  python tests/test_integration.py
"""

import csv
import datetime as dt
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as cli
from gnss.constants import C, FREQ_L1
from gnss.coords import ecef_to_enu_matrix, geodetic_to_ecef
from gnss.ephemeris import GpsEphemeris, sat_pos_vel_clock
from gnss.positioning import earth_rotation_correction
from gnss.timeutils import GPS_EPOCH, gps_to_utc

LAMBDA_L1 = C / FREQ_L1
WEEK, TOW0 = 2405, 302400.0
N_EPOCHS = 15


def _make_constellation() -> dict[str, GpsEphemeris]:
    """8 GPS satellites spread across the sky via varied orbital elements."""
    ephs = {}
    for k in range(8):
        prn = f"G{k + 1:02d}"
        ephs[prn] = GpsEphemeris(
            sat=prn, gps_week=WEEK, toc=TOW0, toe=TOW0,
            af0=1e-4 * (k - 4), af1=0.0, af2=0.0,       # distinct clock offsets
            sqrt_a=5153.65, e=0.004 + 0.001 * k,
            m0=k * (2 * math.pi / 8),
            delta_n=4.5e-9,
            omega0=-math.pi + k * (2 * math.pi / 8),
            i0=0.95 + 0.02 * (k % 3),
            omega=0.3 * k, omega_dot=-8.0e-9, idot=1.0e-10,
            cuc=1e-6, cus=8e-6, crc=250.0, crs=-20.0, cic=-1e-7, cis=1e-7,
            tgd=-5e-9,
        )
    return ephs


def _truth_position(k: int) -> np.ndarray:
    """Receiver moving ~10 m/s east + 2 m/s north from a start point."""
    lat, lon, alt = 32.10, 34.85, 45.0
    east, north = 10.0 * k, 2.0 * k
    enu_to_ecef = ecef_to_enu_matrix(lat, lon).T
    return geodetic_to_ecef(lat, lon, alt) + enu_to_ecef @ np.array([east, north, 0.0])


def _consistent_pseudorange(eph, tow, rcv, cdt_rcv):
    """Iterate to a pseudorange consistent with the solver's forward model.

    Mirrors solver._prepare_satellite exactly: the transmit time is refined once
    by the satellite clock offset (t_tx = tow - pr/c - dts) before the orbit is
    evaluated, so the generated data is a true fixed point of the solver.
    """
    pr = 22_000_000.0
    for _ in range(8):
        t_tx = tow - pr / C
        _, _, dts, _ = sat_pos_vel_clock(eph, t_tx)
        t_tx = tow - pr / C - dts
        pos, vel, dts, _ = sat_pos_vel_clock(eph, t_tx)
        pos = np.array(pos)
        travel = np.linalg.norm(pos - rcv) / C
        pos_rot = earth_rotation_correction(pos, travel)
        rng = np.linalg.norm(pos_rot - rcv)
        pr = rng + cdt_rcv - C * (dts - eph.tgd)
    return pr, np.array(pos), np.array(vel), dts


def _fmt_obs(v):
    return f"{v:14.3f}  " if v is not None else " " * 16


def _fmt_d(v):
    return f"{v: .12E}".rjust(19)


def _write_rinex(obs_path, nav_path, ephs):
    base = GPS_EPOCH + dt.timedelta(seconds=WEEK * 604800 + TOW0)
    truth = []

    # ---- observation file ----
    obs_lines = [
        "     4.01           OBSERVATION DATA    M                   RINEX VERSION / TYPE",
        "G    3 C1C D1C S1C                                          SYS / # / OBS TYPES",
        "  2026    03    21    15    14   52.0000000     GPS         TIME OF FIRST OBS",
        "                                                            END OF HEADER",
    ]
    for k in range(N_EPOCHS):
        tow = TOW0 + k
        t = base + dt.timedelta(seconds=k)
        rcv = _truth_position(k)
        truth.append((gps_to_utc(t), rcv))
        recs = []
        for prn, eph in ephs.items():
            pr, spos, svel, dts = _consistent_pseudorange(eph, tow, rcv, cdt_rcv=0.0)
            los = (spos - rcv) / np.linalg.norm(spos - rcv)
            v_rcv = _truth_position(k + 1) - _truth_position(k)   # constant 10.2 m/s
            range_rate = los @ (svel - v_rcv)
            doppler = -range_rate / LAMBDA_L1
            recs.append(f"{prn}{_fmt_obs(pr)}{_fmt_obs(doppler)}{_fmt_obs(42.0)}")
        stamp = f"> {t.year} {t.month:02d} {t.day:02d} {t.hour:02d} {t.minute:02d} {t.second + t.microsecond/1e6:10.7f}  0 {len(recs):2d}"
        obs_lines.append(stamp)
        obs_lines.extend(recs)
    with open(obs_path, "w") as fh:
        fh.write("\n".join(obs_lines) + "\n")

    # ---- navigation file ----
    nav_lines = [
        "     4.01           N: GNSS NAV DATA    G: GPS              RINEX VERSION / TYPE",
        "                                                            END OF HEADER",
    ]
    for prn, e in ephs.items():
        nav_lines.append(f"> EPH {prn} LNAV")
        epoch_prefix = f"{prn} 2026 03 21 12 00 00"
        nav_lines.append(epoch_prefix + _fmt_d(e.af0) + _fmt_d(e.af1) + _fmt_d(e.af2))
        orbit = [
            [0.0, e.crs, e.delta_n, e.m0],
            [e.cuc, e.e, e.cus, e.sqrt_a],
            [e.toe, e.cic, e.omega0, e.cis],
            [e.i0, e.crc, e.omega, e.omega_dot],
            [e.idot, 0.0, float(e.gps_week), 0.0],
            [2.0, e.health, e.tgd, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
        for row in orbit:
            nav_lines.append("    " + "".join(_fmt_d(v) for v in row))
    with open(nav_path, "w") as fh:
        fh.write("\n".join(nav_lines) + "\n")

    return truth


def test_end_to_end():
    ephs = _make_constellation()
    with tempfile.TemporaryDirectory() as tmp:
        obs_path = os.path.join(tmp, "obs.rnx")
        nav_path = os.path.join(tmp, "nav.rnx")
        out = os.path.join(tmp, "track")
        truth = _write_rinex(obs_path, nav_path, ephs)

        # idealised ranges carry no atmosphere, so solve without the models
        rc = cli.main([obs_path, "--nav", nav_path, "-o", out,
                       "--no-corrections", "--no-spoof-check"])
        assert rc == 0

        with open(out + ".csv") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == N_EPOCHS, len(rows)

        max_err = 0.0
        max_verr = 0.0
        for row, (utc, rcv) in zip(rows, truth):
            est = geodetic_to_ecef(float(row["lat_deg"]), float(row["lon_deg"]),
                                   float(row["alt_m"]))
            max_err = max(max_err, np.linalg.norm(est - rcv))
            # truth velocity is ~10.2 m/s (east+north); check speed is close
            max_verr = max(max_verr, abs(float(row["speed_ms"]) - math.hypot(10.0, 2.0)))
        assert max_err < 1e-2, f"position error {max_err:.4f} m"
        assert max_verr < 1e-1, f"velocity error {max_verr:.4f} m/s"
        print(f"ok  end-to-end: {N_EPOCHS} epochs, "
              f"max pos err={max_err*1e3:.3f} mm, max speed err={max_verr*1e3:.3f} mm/s")


if __name__ == "__main__":
    test_end_to_end()
    print("\nIntegration test passed.")
