#!/usr/bin/env python3
"""Compute a 1 Hz GNSS trajectory from a RINEX observation file.

Usage:
    python main.py OBS.rnx --nav NAV.rnx -o output/track
    python main.py OBS.rnx -o output/track          # auto-download ephemeris

Outputs <out>.csv and <out>.kml with position, velocity and UTC time per epoch.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from gnss import output
from gnss.ephemeris_download import download_brdc
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
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

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
    print(f"      ephemeris for {len(nav)} GPS satellites")

    print(f"[3/4] Solving {len(epochs)} epochs (GPS-only)...")
    solutions = []
    prev_pos = None
    for ep in epochs:
        sol = solve_epoch(ep, nav, prev_pos=prev_pos, min_sats=args.min_sats)
        if sol is not None:
            solutions.append(sol)
            prev_pos = np.append(sol.ecef, sol.clock_bias_m)

    if not solutions:
        print("      No epochs could be solved. Check that the nav file covers "
              "the observation time and contains GPS ephemeris.", file=sys.stderr)
        return 1
    print(f"      solved {len(solutions)}/{len(epochs)} epochs")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    csv_path, kml_path = args.out + ".csv", args.out + ".kml"
    output.write_csv(csv_path, solutions)
    output.write_kml(kml_path, solutions)
    print(f"[4/4] Wrote {csv_path} and {kml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
