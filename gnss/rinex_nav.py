"""Parser for RINEX 3.x / 4.x NAVIGATION files (GPS broadcast ephemeris).

A GPS navigation record is 8 lines: one epoch/clock line followed by seven
"broadcast orbit" lines, each holding up to four numbers in Fortran D19.12
fields.  RINEX 4 additionally prefixes every record with a '> EPH G01 LNAV'
marker line, which we simply skip.

Because negative values in D19.12 can butt up against the previous field with
no separating space, we extract numbers by fixed column position rather than
by ``str.split()``.
"""

from __future__ import annotations

from .ephemeris import GpsEphemeris

# Column start positions (0-based) of the D19.12 fields.
_EPOCH_LINE_COLS = (23, 42, 61)          # 3 clock terms after the 23-char prefix
_ORBIT_LINE_COLS = (4, 23, 42, 61)       # 4 values, 4-space indent
_FIELD_WIDTH = 19


def _read_float(line: str, start: int) -> float:
    """Read one D19.12 number; Fortran 'D' exponent -> Python 'E'. Blank -> 0.0."""
    raw = line[start:start + _FIELD_WIDTH].strip()
    if not raw:
        return 0.0
    return float(raw.replace("D", "E").replace("d", "e"))


def _orbit_values(line: str) -> list[float]:
    return [_read_float(line, c) for c in _ORBIT_LINE_COLS]


def _parse_gps_record(lines: list[str]) -> GpsEphemeris:
    """Build a :class:`GpsEphemeris` from its 8 text lines."""
    l0 = lines[0]
    sat = l0[:3].strip()
    if len(sat) == 2:                      # normalise 'G8' -> 'G08'
        sat = sat[0] + "0" + sat[1]

    # epoch line: 3 clock coefficients (toc is derived later from the epoch fields)
    af0, af1, af2 = (_read_float(l0, c) for c in _EPOCH_LINE_COLS)

    o1 = _orbit_values(lines[1])   # IODE, Crs, Delta_n, M0
    o2 = _orbit_values(lines[2])   # Cuc, e, Cus, sqrtA
    o3 = _orbit_values(lines[3])   # Toe, Cic, OMEGA0, Cis
    o4 = _orbit_values(lines[4])   # i0, Crc, omega, OMEGADOT
    o5 = _orbit_values(lines[5])   # IDOT, codesL2, GPSWeek, L2Pflag
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


def parse_nav(path: str) -> dict[str, list[GpsEphemeris]]:
    """Parse a navigation file -> {sat_id: [GpsEphemeris, ...]} (GPS only for now)."""
    ephemerides: dict[str, list[GpsEphemeris]] = {}

    with open(path, "r", errors="replace") as fh:
        # skip header
        for line in fh:
            if line[60:].rstrip() == "END OF HEADER":
                break

        pending: list[str] = []
        current_sys = None
        for line in fh:
            if line.startswith(">"):          # RINEX 4 record marker: '> EPH G01 LNAV'
                current_sys = line.split()[2][0] if len(line.split()) > 2 else None
                pending = []
                continue
            if not line.strip():
                continue

            # A data line whose first column is a system letter begins a record.
            if line[0].isalpha() and not pending:
                current_sys = line[0]

            pending.append(line.rstrip("\n"))

            if len(pending) == 8:             # complete GPS record collected
                if (current_sys or pending[0][0]) == "G":
                    eph = _parse_gps_record(pending)
                    ephemerides.setdefault(eph.sat, []).append(eph)
                pending = []
                current_sys = None

    return ephemerides
