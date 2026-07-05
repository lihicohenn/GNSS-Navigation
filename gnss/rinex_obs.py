"""Parser for RINEX 3.x / 4.x OBSERVATION files (Android GnssLogger output).

We only depend on the parts of the spec that GnssLogger actually emits, and we
parse defensively (blank fields = missing measurement).  The output is a plain
list of :class:`Epoch` objects that the positioning code can iterate over.

Observation record layout (RINEX 3+):
    - epoch line starts with '>' and carries the calendar time + satellite count
    - each following line is  <3-char sat id> then N observations
    - every observation occupies 16 columns: an F14.3 value followed by two
      single-character flags (LLI, SSI).  A blank value means "not measured".
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field


@dataclass
class RinexObsHeader:
    version: float
    file_type: str
    system: str                              # 'M' = mixed, 'G' = GPS only, ...
    obs_types: dict[str, list[str]] = field(default_factory=dict)  # sys -> codes
    time_first_obs: dt.datetime | None = None
    time_system: str = "GPS"


@dataclass
class Epoch:
    time: dt.datetime                        # GPS time scale, tz-aware
    flag: int
    sats: dict[str, dict[str, float]]        # 'G08' -> {'C1C': 2.1e7, 'D1C': ...}


_OBS_FIELD_WIDTH = 16   # 14 (value) + 2 (LLI, SSI)
_VALUE_WIDTH = 14


def _parse_obs_types(line: str, count: int, fh) -> list[str]:
    """Read `count` observation codes, following continuation lines if needed."""
    codes = line[7:60].split()
    while len(codes) < count:
        codes += next(fh)[7:60].split()
    return codes[:count]


def parse_header(fh) -> RinexObsHeader:
    """Consume the header from an open file handle, up to END OF HEADER."""
    first = next(fh)
    version = float(first[:9])
    header = RinexObsHeader(
        version=version,
        file_type=first[20:21],
        system=first[40:41],
    )

    for line in fh:
        label = line[60:].rstrip()
        if label == "SYS / # / OBS TYPES":
            sys_char = line[0]
            count = int(line[3:6])
            header.obs_types[sys_char] = _parse_obs_types(line, count, fh)
        elif label == "TIME OF FIRST OBS":
            parts = line[:60].split()
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            hour, minute = int(parts[3]), int(parts[4])
            sec = float(parts[5])
            header.time_system = parts[6] if len(parts) > 6 else "GPS"
            header.time_first_obs = dt.datetime(
                year, month, day, hour, minute, int(sec),
                int((sec % 1) * 1e6), tzinfo=dt.timezone.utc,
            )
        elif label == "END OF HEADER":
            break

    return header


def _parse_epoch_line(line: str) -> tuple[dt.datetime, int, int]:
    """Parse a '>' epoch header line -> (time, epoch_flag, num_sats)."""
    p = line[1:].split()
    year, month, day = int(p[0]), int(p[1]), int(p[2])
    hour, minute = int(p[3]), int(p[4])
    sec = float(p[5])
    flag = int(p[6])
    nsat = int(p[7])
    t = dt.datetime(
        year, month, day, hour, minute, int(sec),
        round((sec % 1) * 1e6), tzinfo=dt.timezone.utc,
    )
    return t, flag, nsat


def _parse_obs_record(line: str, codes: list[str]) -> tuple[str, dict[str, float]]:
    """Parse one satellite's observation line -> (sat_id, {code: value})."""
    sat_id = line[:3].strip()
    # GnssLogger writes 'G8' occasionally; normalise to zero-padded 'G08'.
    if len(sat_id) == 2:
        sat_id = sat_id[0] + "0" + sat_id[1]

    values: dict[str, float] = {}
    for k, code in enumerate(codes):
        start = 3 + k * _OBS_FIELD_WIDTH
        raw = line[start:start + _VALUE_WIDTH].strip()
        if raw:
            try:
                values[code] = float(raw)
            except ValueError:
                # GnssLogger occasionally writes an over-wide value that spills
                # past its column; skip the unparseable field rather than crash.
                pass
    return sat_id, values


def parse_obs(path: str) -> tuple[RinexObsHeader, list[Epoch]]:
    """Parse a full RINEX observation file into (header, epochs)."""
    with open(path, "r", errors="replace") as fh:
        header = parse_header(fh)
        epochs: list[Epoch] = []

        for line in fh:
            if not line.strip() or not line.startswith(">"):
                continue
            time, flag, nsat = _parse_epoch_line(line)
            sats: dict[str, dict[str, float]] = {}
            for _ in range(nsat):
                rec = next(fh).rstrip("\n")
                if not rec.strip():
                    continue
                sat_id, values = _parse_obs_record(rec, header.obs_types.get(rec[0], []))
                if values:
                    sats[sat_id] = values
            epochs.append(Epoch(time=time, flag=flag, sats=sats))

    return header, epochs
