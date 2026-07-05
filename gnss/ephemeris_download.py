"""Fallback: download broadcast ephemeris when no local nav file is provided.

A RINEX *observation* file has no orbits, so if the shared folder gave us only
observations we fetch the daily merged *mixed* broadcast navigation file (BRDC
..._MN) for the recording date from a public IGS mirror.  This keeps the
solution based on the RINEX measurements while sourcing orbits from the standard
broadcast product.

If every automatic mirror fails (offline, mirror down), the printed URLs let you
download the file by hand and pass it with ``--nav``.
"""

from __future__ import annotations

import datetime as dt
import glob
import gzip
import os
import urllib.request

# Public IGS mirrors that serve the daily merged mixed-GNSS broadcast file.
# Each entry is a full URL template; note the product name differs by archive
# (BKG serves BRDC00WRD, the IGS combination is BRDC00IGS).  {yy} is the 2-digit
# year.  BKG is listed first: it is reliable and needs no login.
_MIRRORS = [
    "https://igs.bkg.bund.de/root_ftp/IGS/BRDC/{yyyy}/{ddd}/BRDC00WRD_R_{yyyy}{ddd}0000_01D_MN.rnx.gz",
    "https://cddis.nasa.gov/archive/gnss/data/daily/{yyyy}/{ddd}/{yy}p/BRDC00IGS_R_{yyyy}{ddd}0000_01D_MN.rnx.gz",
    "https://igs.ign.fr/pub/igs/data/{yyyy}/{ddd}/BRDC00IGS_R_{yyyy}{ddd}0000_01D_MN.rnx.gz",
]

# A browser-like User-Agent avoids the odd mirror that rejects urllib's default.
_HEADERS = {"User-Agent": "Mozilla/5.0 (GNSS-Navigation ephemeris fetch)"}


def brdc_filename(date: dt.date) -> str:
    """Canonical local filename for the merged nav file of a given date."""
    doy = date.timetuple().tm_yday
    return f"BRDC00IGS_R_{date.year}{doy:03d}0000_01D_MN.rnx"


def download_brdc(date: dt.date, dest_dir: str) -> str:
    """Download + gunzip the daily broadcast nav file. Returns the local path.

    Raises RuntimeError with a helpful message if every mirror fails.
    """
    os.makedirs(dest_dir, exist_ok=True)
    doy = f"{date.timetuple().tm_yday:03d}"

    # reuse any previously downloaded merged nav file for this date
    cached = glob.glob(os.path.join(dest_dir, f"*_{date.year}{doy}0000_01D_MN.rnx"))
    if cached:
        return cached[0]

    errors = []
    for template in _MIRRORS:
        url = template.format(yyyy=date.year, ddd=doy, yy=f"{date.year % 100:02d}")
        out_path = os.path.join(dest_dir, os.path.basename(url)[:-3])   # strip .gz
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=45) as resp:
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
