"""Cross-check our RINEX-only trajectory against the phone's NMEA fix.

This is a *validation*, not an input: for every computed epoch we find the NMEA
fix closest in time and measure how far apart the two positions are, in the
local horizontal (East-North) plane and vertically.  The receiver's own solution
is not error-free either, so this quantifies agreement rather than absolute
truth — but a well-behaved SPP solution should sit within a few metres of it.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

from .coords import ecef_to_enu_matrix, geodetic_to_ecef
from .nmea import NmeaFix
from .solver import EpochSolution

_SECONDS_PER_DAY = 86400.0


@dataclass
class ValidationReport:
    n_matched: int
    n_solutions: int
    horiz_mean: float
    horiz_rms: float
    horiz_median: float
    horiz_p95: float
    horiz_max: float
    vert_rms: float
    horiz_errors: list[float]      # per matched epoch, for plotting/inspection


def _tod_utc(solution: EpochSolution) -> float:
    t = solution.time_utc
    return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6


def _nearest(fix_tods: list[float], fixes: list[NmeaFix], tod: float,
             tol: float) -> NmeaFix | None:
    """Nearest NMEA fix in time (handles wrap around midnight), within ``tol`` s."""
    if not fix_tods:
        return None
    i = bisect.bisect_left(fix_tods, tod)
    best, best_dt = None, tol
    for j in (i - 1, i):
        if 0 <= j < len(fix_tods):
            dt = abs(fix_tods[j] - tod)
            dt = min(dt, _SECONDS_PER_DAY - dt)     # midnight wrap
            if dt <= best_dt:
                best, best_dt = fixes[j], dt
    return best


def compare(solutions: list[EpochSolution], fixes: list[NmeaFix],
            tol_s: float = 0.75) -> ValidationReport:
    """Match each solution to the nearest NMEA fix and summarise the differences."""
    fix_tods = [f.tod for f in fixes]

    horiz, vert = [], []
    for s in solutions:
        fix = _nearest(fix_tods, fixes, _tod_utc(s), tol_s)
        if fix is None:
            continue
        ref = geodetic_to_ecef(fix.lat, fix.lon, s.alt)   # same height -> pure horizontal
        enu = ecef_to_enu_matrix(fix.lat, fix.lon) @ (s.ecef - ref)
        horiz.append(float(np.hypot(enu[0], enu[1])))
        ref_alt = fix.alt_ellipsoidal
        if ref_alt is not None:
            vert.append(s.alt - ref_alt)

    h = np.array(horiz) if horiz else np.array([np.nan])
    v = np.array(vert) if vert else np.array([np.nan])
    return ValidationReport(
        n_matched=len(horiz),
        n_solutions=len(solutions),
        horiz_mean=float(np.mean(h)),
        horiz_rms=float(np.sqrt(np.mean(h**2))),
        horiz_median=float(np.median(h)),
        horiz_p95=float(np.percentile(h, 95)),
        horiz_max=float(np.max(h)),
        vert_rms=float(np.sqrt(np.mean(v**2))),
        horiz_errors=horiz,
    )


def format_report(report: ValidationReport) -> str:
    """Human-readable summary for the console."""
    if report.n_matched == 0:
        return ("      NMEA validation: no epochs matched in time — check that the "
                "NMEA file covers the same recording.")
    return (
        f"      NMEA validation: matched {report.n_matched}/{report.n_solutions} epochs\n"
        f"        horizontal error  mean={report.horiz_mean:.2f} m  "
        f"rms={report.horiz_rms:.2f} m  median={report.horiz_median:.2f} m\n"
        f"        horizontal error  p95={report.horiz_p95:.2f} m  "
        f"max={report.horiz_max:.2f} m\n"
        f"        vertical error    rms={report.vert_rms:.2f} m"
    )
