#!/usr/bin/env python3
"""Compute a 1 Hz GNSS trajectory from a RINEX observation file.

Usage:
    python main.py OBS.rnx --nav NAV.rnx -o output/track
    python main.py OBS.rnx -o output/track            # auto-download ephemeris
    python main.py OBS.rnx --nav NAV.rnx --nmea phone.nmea   # + validation
    python main.py OBS.rnx --nav NAV.rnx --systems G         # GPS-only

Outputs <out>.csv and <out>.kml with position, velocity and UTC time per epoch.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from gnss import output, spoofing, validate
from gnss.coords import ecef_to_enu_matrix
from gnss.ephemeris_download import download_brdc
from gnss.nmea import parse_nmea
from gnss.rinex_nav import parse_nav
from gnss.rinex_obs import parse_obs
from gnss.solver import solve_epoch


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("obs", help="RINEX observation file (.rnx / .obs / .YYo)")
    p.add_argument("--nav", help="RINEX navigation file. If omitted, broadcast "
                                 "ephemeris is downloaded for the recording date.")
    p.add_argument("-o", "--out", default="output/track",
                   help="output path prefix (default: output/track)")
    p.add_argument("--min-sats", type=int, default=4,
                   help="minimum satellites required for a fix (default 4)")
    p.add_argument("--systems", default=None,
                   help="restrict constellations, e.g. 'G', 'GE', 'GEC' "
                        "(default: use every system available)")
    p.add_argument("--no-corrections", action="store_true",
                   help="disable ionospheric/tropospheric corrections and the "
                        "elevation mask (useful for idealised/synthetic data)")
    p.add_argument("--elevation-mask", type=float, default=5.0,
                   help="elevation mask in degrees (default 5)")
    p.add_argument("--max-speed", type=float, default=60.0,
                   help="reject fixes implying a speed above this m/s vs the "
                        "previous fix (default 60 m/s ~ 216 km/h)")
    p.add_argument("--nmea", help="phone NMEA file to validate the solution against")
    p.add_argument("--no-spoof-check", action="store_true",
                   help="skip the spoofing/integrity analysis")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    corrections = not args.no_corrections

    print(f"[1/4] Parsing observations: {args.obs}")
    header, epochs = parse_obs(args.obs)
    print(f"      {len(epochs)} epochs, systems: {sorted(header.obs_types)}")

    nav_path = args.nav
    if nav_path is None:
        date = header.time_first_obs.date()
        print(f"[2/4] No --nav given; downloading broadcast ephemeris for {date}")
        nav_path = download_brdc(date, dest_dir="data")
    print(f"[2/4] Parsing navigation: {nav_path}")
    nav = parse_nav(nav_path)
    iono = "yes" if nav.iono_alpha else "no"
    print(f"      ephemeris for {len(nav)} satellites; Klobuchar iono coeffs: {iono}")

    mode = "all systems" if args.systems is None else args.systems
    print(f"[3/4] Solving {len(epochs)} epochs ({mode}, "
          f"corrections {'on' if corrections else 'off'})...")
    solutions = []
    prev_pos = None
    prev_ecef = None
    prev_time = None
    n_jumps = 0
    for ep in epochs:
        sol = solve_epoch(
            ep, nav, prev_pos=prev_pos, min_sats=args.min_sats,
            corrections=corrections, elevation_mask_deg=args.elevation_mask,
            systems=args.systems,
        )
        if sol is None:
            continue

        # kinematic plausibility: drop a fix that implies an impossible speed from
        # the previous good fix (a residual outlier the geometry could not catch).
        if prev_ecef is not None:
            dt = (ep.time - prev_time).total_seconds()
            enu = ecef_to_enu_matrix(sol.lat, sol.lon) @ (sol.ecef - prev_ecef)
            horiz = float(np.hypot(enu[0], enu[1]))
            if dt > 0 and horiz / dt > args.max_speed:
                n_jumps += 1
                continue

        solutions.append(sol)
        prev_pos = np.append(sol.ecef, sol.clock_bias_m)
        prev_ecef, prev_time = sol.ecef, ep.time

    if not solutions:
        print("      No epochs could be solved. Check that the nav file covers "
              "the observation time and contains ephemeris.", file=sys.stderr)
        return 1
    used = sorted({s for sol in solutions for s in sol.systems})
    jump_note = f" ({n_jumps} kinematic outliers dropped)" if n_jumps else ""
    print(f"      solved {len(solutions)}/{len(epochs)} epochs "
          f"using {''.join(used)}{jump_note}")

    # ---- optional NMEA validation ----
    reference = None
    if args.nmea:
        fixes = parse_nmea(args.nmea)
        print(f"      loaded {len(fixes)} NMEA fixes from {args.nmea}")
        report = validate.compare(solutions, fixes)
        print(validate.format_report(report))
        reference = [(f.lat, f.lon) for f in fixes]

    # ---- optional spoofing / integrity analysis ----
    if not args.no_spoof_check:
        print(spoofing.format_report(spoofing.analyze(solutions)))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    csv_path, kml_path = args.out + ".csv", args.out + ".kml"
    output.write_csv(csv_path, solutions)
    output.write_kml(kml_path, solutions, reference=reference)
    print(f"[4/4] Wrote {csv_path} and {kml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
