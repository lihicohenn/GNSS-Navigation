"""Fallback: download broadcast ephemeris when no local nav file is provided.

A RINEX *observation* file has no orbits, so if the shared folder gave us only
observations we fetch the daily merged broadcast navigation file (BRDC) for the
recording date from a public IGS mirror.  This keeps the solution based on the
RINEX measurements while sourcing orbits from the standard broadcast product.

If the automatic download fails (offline, mirror down), the printed URL lets you
download the file by hand and pass it with ``--nav``.
"""

from __future__ import annotations

import datetime as dt
import gzip
import os
import urllib.request

# Public, no-login IGS data mirror (IGN, France). {yyyy}=year, {ddd}=day-of-year.
_MIRRORS = [
    "https://igs.ign.fr/pub/igs/data/{yyyy}/{ddd}/BRDC00IGS_R_{yyyy}{ddd}0000_01D_MN.rnx.gz",
    "https://gssc.esa.int/gnss/data/daily/{yyyy}/{ddd}/BRDC00IGS_R_{yyyy}{ddd}0000_01D_MN.rnx.gz",
]


def brdc_filename(date: dt.date) -> str:
    doy = date.timetuple().tm_yday
    return f"BRDC00IGS_R_{date.year}{doy:03d}0000_01D_MN.rnx"


def download_brdc(date: dt.date, dest_dir: str) -> str:
    """Download + gunzip the daily broadcast nav file. Returns the local path.

    Raises RuntimeError with a helpful message if every mirror fails.
    """
    os.makedirs(dest_dir, exist_ok=True)
    doy = date.timetuple().tm_yday
    out_path = os.path.join(dest_dir, brdc_filename(date))
    if os.path.exists(out_path):
        return out_path

    errors = []
    for template in _MIRRORS:
        url = template.format(yyyy=date.year, ddd=f"{doy:03d}")
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = gzip.decompress(resp.read())
            with open(out_path, "wb") as fh:
                fh.write(data)
            return out_path
        except Exception as exc:  # noqa: BLE001 - report and try next mirror
            errors.append(f"  {url}\n    -> {exc}")

    raise RuntimeError(
        "Could not download broadcast ephemeris automatically. Tried:\n"
        + "\n".join(errors)
        + "\n\nDownload the file manually and re-run with --nav <file>."
    )
