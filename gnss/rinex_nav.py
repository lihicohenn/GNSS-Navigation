"""Parser for RINEX 3.x / 4.x NAVIGATION files (multi-GNSS broadcast ephemeris).

Two record shapes are handled:

* **Keplerian** (GPS 'G', Galileo 'E', BeiDou 'C', QZSS 'J') — 8 lines: one
  epoch/clock line followed by seven "broadcast orbit" lines, each holding up to
  four numbers in Fortran D19.12 fields.
* **GLONASS** ('R') — 4 lines: an epoch/clock line plus three lines carrying the
  PZ-90 position, velocity and acceleration state vector.

RINEX 4 additionally prefixes every ephemeris record with a ``> EPH G01 LNAV``
marker line and can carry ionosphere coefficients in ``> ION`` records; RINEX 3
keeps ionosphere in the header (``IONOSPHERIC CORR``).  Both are handled.

Because negative values in D19.12 can butt up against the previous field with no
separating space, numbers are extracted by fixed column position, not by
``str.split()``.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

from . import timeutils
from .constants import GPS_UTC_LEAP_SECONDS
from .ephemeris import GlonassEphemeris, GpsEphemeris

# Column start positions (0-based) of the D19.12 fields.
_EPOCH_LINE_COLS = (23, 42, 61)          # 3 clock terms after the 23-char prefix
_ORBIT_LINE_COLS = (4, 23, 42, 61)       # 4 values, 4-space indent
_FIELD_WIDTH = 19

_KEPLER_SYS = ("G", "E", "C", "J")       # constellations using Keplerian records


@dataclass
class NavData:
    """Parsed navigation data: broadcast ephemeris + ionosphere model."""

    eph: dict[str, list] = field(default_factory=dict)   # sat -> [ephemeris, ...]
    iono_alpha: tuple | None = None      # Klobuchar alpha coefficients (4)
    iono_beta: tuple | None = None       # Klobuchar beta coefficients (4)

    def __len__(self) -> int:
        return len(self.eph)


def _read_float(line: str, start: int) -> float:
    """Read one D19.12 number; Fortran 'D' exponent -> Python 'E'. Blank -> 0.0."""
    raw = line[start:start + _FIELD_WIDTH].strip()
    if not raw:
        return 0.0
    return float(raw.replace("D", "E").replace("d", "e"))


def _orbit_values(line: str) -> list[float]:
    return [_read_float(line, c) for c in _ORBIT_LINE_COLS]


def _epoch_datetime(line: str) -> dt.datetime:
    """Parse the calendar epoch (UTC) from a navigation record's first line."""
    p = line[3:23].split()
    year, month, day = int(p[0]), int(p[1]), int(p[2])
    hour, minute, sec = int(p[3]), int(p[4]), int(float(p[5]))
    return dt.datetime(year, month, day, hour, minute, sec, tzinfo=dt.timezone.utc)


def _normalise_sat(raw: str) -> str:
    sat = raw.strip()
    if len(sat) == 2:                      # 'G8' -> 'G08'
        sat = sat[0] + "0" + sat[1]
    return sat


def _parse_kepler_record(lines: list[str]) -> GpsEphemeris:
    """Build a Keplerian :class:`GpsEphemeris` from its 8 text lines (G/E/C/J)."""
    l0 = lines[0]
    sat = _normalise_sat(l0[:3])

    # epoch line: 3 clock coefficients (toc is derived later from the epoch fields)
    af0, af1, af2 = (_read_float(l0, c) for c in _EPOCH_LINE_COLS)

    o1 = _orbit_values(lines[1])   # IODE, Crs, Delta_n, M0
    o2 = _orbit_values(lines[2])   # Cuc, e, Cus, sqrtA
    o3 = _orbit_values(lines[3])   # Toe, Cic, OMEGA0, Cis
    o4 = _orbit_values(lines[4])   # i0, Crc, omega, OMEGADOT
    o5 = _orbit_values(lines[5])   # IDOT, codesL2, Week, L2Pflag
    o6 = _orbit_values(lines[6])   # SVaccuracy, SVhealth, TGD, IODC

    toe = o3[0]
    gps_week = int(o5[2])

    return GpsEphemeris(
        sat=sat,
        gps_week=gps_week,
        toc=toe,           # toc set to toe seconds-of-week (same week); good enough here
        toe=toe,
        af0=af0, af1=af1, af2=af2,
        crs=o1[1], delta_n=o1[2], m0=o1[3],
        cuc=o2[0], e=o2[1], cus=o2[2], sqrt_a=o2[3],
        cic=o3[1], omega0=o3[2], cis=o3[3],
        i0=o4[0], crc=o4[1], omega=o4[2], omega_dot=o4[3],
        idot=o5[0],
        health=o6[1], tgd=o6[2],
    )


# Backward-compatible alias (the record layout is identical for all Keplerian
# systems; the historical name is kept for the self-tests).
_parse_gps_record = _parse_kepler_record


def _parse_glonass_record(lines: list[str]) -> GlonassEphemeris:
    """Build a :class:`GlonassEphemeris` from its 4 text lines.

    The epoch is in UTC, so it is converted to GPS seconds-of-week to match the
    time scale used by the rest of the pipeline.
    """
    l0 = lines[0]
    sat = _normalise_sat(l0[:3])

    tau_n, gamma_n, _msg_time = (_read_float(l0, c) for c in _EPOCH_LINE_COLS)

    o1 = _orbit_values(lines[1])   # X, Xdot, Xacc, health
    o2 = _orbit_values(lines[2])   # Y, Ydot, Yacc, frequency number
    o3 = _orbit_values(lines[3])   # Z, Zdot, Zacc, age of info

    # UTC epoch -> GPS seconds of week (GPS time leads UTC by the leap-second count)
    gps_epoch = _epoch_datetime(l0) + dt.timedelta(seconds=GPS_UTC_LEAP_SECONDS)
    _, tow = timeutils.datetime_to_gps_week_tow(gps_epoch)

    return GlonassEphemeris(
        sat=sat,
        toe=tow, toc=tow,
        tau_n=tau_n, gamma_n=gamma_n,
        pos=np.array([o1[0], o2[0], o3[0]]) * 1e3,     # km -> m
        vel=np.array([o1[1], o2[1], o3[1]]) * 1e3,     # km/s -> m/s
        acc=np.array([o1[2], o2[2], o3[2]]) * 1e3,     # km/s^2 -> m/s^2
        freq_num=int(o2[3]),
        health=o1[3],
    )


def _dispatch_record(block: list[str], sysc: str, nav: NavData) -> None:
    """Parse one ephemeris record and file it under its satellite id."""
    if len(block) < 4 or block[0][:3].strip() == "":
        return
    try:
        if sysc == "R":
            eph = _parse_glonass_record(block)
        elif sysc in _KEPLER_SYS:
            eph = _parse_kepler_record(block)
        else:                              # SBAS / IRNSS etc.: skip cleanly
            return
    except (ValueError, IndexError):
        return                             # tolerate a malformed record
    nav.eph.setdefault(eph.sat, []).append(eph)


def _floats_from(text: str) -> list[float]:
    out = []
    for tok in text.replace("D", "E").replace("d", "e").split():
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def _parse_header_iono(line: str, label: str, nav: NavData) -> None:
    """Extract Klobuchar coefficients from a RINEX 2/3 header line."""
    if label == "IONOSPHERIC CORR":
        key = line[:4].strip()
        vals = _floats_from(line[5:53])
        if key == "GPSA" and len(vals) >= 4:
            nav.iono_alpha = tuple(vals[:4])
        elif key == "GPSB" and len(vals) >= 4:
            nav.iono_beta = tuple(vals[:4])
    elif label == "ION ALPHA":             # RINEX 2
        vals = _floats_from(line[2:50])
        if len(vals) >= 4:
            nav.iono_alpha = tuple(vals[:4])
    elif label == "ION BETA":
        vals = _floats_from(line[2:50])
        if len(vals) >= 4:
            nav.iono_beta = tuple(vals[:4])


def _parse_body_iono(marker: list[str], data: list[str], nav: NavData) -> None:
    """Best-effort Klobuchar extraction from a RINEX 4 ``> ION`` record.

    Only GPS LNAV Klobuchar records (8 coefficients) are used; anything else is
    ignored so we never feed the model coefficients we cannot trust.
    """
    if len(marker) < 2 or marker[1][0] != "G":
        return
    vals = _floats_from(" ".join(data))
    # first token on the first data line is the transmit time-of-week; drop it
    if len(vals) >= 9:
        vals = vals[1:9]
    if len(vals) >= 8 and nav.iono_alpha is None:
        nav.iono_alpha = tuple(vals[:4])
        nav.iono_beta = tuple(vals[4:8])


def parse_nav(path: str) -> NavData:
    """Parse a navigation file into a :class:`NavData` (all constellations)."""
    with open(path, "r", errors="replace") as fh:
        lines = fh.readlines()

    nav = NavData()

    # ---- header: capture ionosphere coefficients, find END OF HEADER ----
    body_start = len(lines)
    for i, line in enumerate(lines):
        label = line[60:].rstrip()
        if label in ("IONOSPHERIC CORR", "ION ALPHA", "ION BETA"):
            _parse_header_iono(line, label, nav)
        elif label == "END OF HEADER":
            body_start = i + 1
            break

    # ---- body ----
    #
    # Records are delimited structurally rather than by a hard-coded line count:
    # an epoch line starts in column 0 with a system letter, and its broadcast-
    # orbit lines that follow are indented (they start with a space).  This is
    # essential because a record's length varies not just by system (GPS 8 lines,
    # GLONASS 4) but by RINEX minor version — RINEX 3.05 adds a 5th line to each
    # GLONASS record (extra group-delay/health parameters).  Counting lines would
    # desynchronise on that extra line and silently drop most GLONASS ephemerides.
    i = body_start
    n_lines = len(lines)

    def gather_indented(start: int) -> int:
        """Index past a RINEX 3 record: epoch line + its indented orbit lines."""
        j = start + 1
        while (j < n_lines and lines[j].strip()
               and not lines[j][0].isalpha() and not lines[j].startswith(">")):
            j += 1
        return j

    def gather_to_marker(start: int) -> int:
        """Index past a RINEX 4 record body: everything up to the next '>' marker."""
        j = start + 1
        while j < n_lines and lines[j].strip() and not lines[j].startswith(">"):
            j += 1
        return j

    while i < n_lines:
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        if line.startswith(">"):                       # RINEX 4 record marker
            toks = line[1:].split()
            rectype = toks[0] if toks else ""
            end = gather_to_marker(i)                   # epoch line + orbit lines
            body = lines[i + 1:end]
            if rectype == "EPH":
                sysc = toks[1][0] if len(toks) > 1 else (body[0][0] if body else "")
                _dispatch_record(body, sysc, nav)
            elif rectype == "ION":
                _parse_body_iono(toks, body, nav)
            i = end
            continue

        # RINEX 3: a data line whose first column is a system letter starts a record
        end = gather_indented(i)
        _dispatch_record(lines[i:end], line[0], nav)
        i = end

    return nav
