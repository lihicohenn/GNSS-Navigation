"""Parser for the phone's own NMEA-0183 fix (for cross-checking only).

Android's GnssLogger can record an NMEA stream alongside the RINEX observations.
That stream is the receiver's *internal* position solution — we never feed it
into our own computation, but it is the natural ground truth to validate our
RINEX-only trajectory against (see :mod:`gnss.validate`).

We read the two sentences that carry a position: ``GGA`` (lat/lon/altitude +
fix quality) and ``RMC`` (lat/lon + date + ground speed).  Only the UTC
*time-of-day* is stored, which is all the validation needs to line the two
tracks up second-by-second.
"""

from __future__ import annotations

from dataclasses import dataclass

_KNOTS_TO_MS = 0.514444


@dataclass
class NmeaFix:
    tod: float                    # UTC seconds of day
    lat: float                    # degrees, +N
    lon: float                    # degrees, +E
    alt: float | None = None      # orthometric altitude (MSL) [m]
    geoid_sep: float | None = None  # geoid-ellipsoid separation [m]
    quality: int = 0              # GGA fix quality (0 = invalid)
    speed: float | None = None    # ground speed [m/s] (from RMC)

    @property
    def alt_ellipsoidal(self) -> float | None:
        """Ellipsoidal height, comparable to our WGS-84 solution."""
        if self.alt is None:
            return None
        return self.alt + (self.geoid_sep or 0.0)


def _dm_to_deg(value: str, hemi: str) -> float | None:
    """Convert an NMEA 'ddmm.mmmm' / 'dddmm.mmmm' angle to signed degrees."""
    if not value:
        return None
    v = float(value)
    deg = int(v // 100)
    minutes = v - deg * 100
    dec = deg + minutes / 60.0
    if hemi in ("S", "W"):
        dec = -dec
    return dec


def _tod(hms: str) -> float | None:
    """'hhmmss.ss' -> seconds of day."""
    if not hms or len(hms) < 6:
        return None
    return int(hms[0:2]) * 3600 + int(hms[2:4]) * 60 + float(hms[4:])


def _checksum_ok(sentence: str) -> bool:
    """Validate the '*HH' XOR checksum if present; accept sentences without one."""
    if "*" not in sentence:
        return True
    body, _, cs = sentence.partition("*")
    body = body.lstrip("$")
    want = 0
    for ch in body:
        want ^= ord(ch)
    try:
        return want == int(cs[:2], 16)
    except ValueError:
        return False


def _parse_gga(f: list[str]) -> NmeaFix | None:
    tod = _tod(f[1])
    lat = _dm_to_deg(f[2], f[3])
    lon = _dm_to_deg(f[4], f[5])
    if tod is None or lat is None or lon is None:
        return None
    quality = int(f[6]) if f[6] else 0
    alt = float(f[9]) if len(f) > 9 and f[9] else None
    geoid = float(f[11]) if len(f) > 11 and f[11] else None
    return NmeaFix(tod=tod, lat=lat, lon=lon, alt=alt, geoid_sep=geoid, quality=quality)


def _parse_rmc(f: list[str]) -> NmeaFix | None:
    if len(f) < 8 or f[2] != "A":                # 'A' = valid, 'V' = warning
        return None
    tod = _tod(f[1])
    lat = _dm_to_deg(f[3], f[4])
    lon = _dm_to_deg(f[5], f[6])
    if tod is None or lat is None or lon is None:
        return None
    speed = float(f[7]) * _KNOTS_TO_MS if f[7] else None
    return NmeaFix(tod=tod, lat=lat, lon=lon, quality=1, speed=speed)


def parse_nmea(path: str) -> list[NmeaFix]:
    """Parse an NMEA file into a time-ordered list of fixes (GGA preferred).

    Where both a GGA and an RMC exist for the same instant, the GGA (which has
    altitude) wins but the RMC's ground speed is merged in.
    """
    by_tod: dict[float, NmeaFix] = {}
    with open(path, "r", errors="replace") as fh:
        for raw in fh:
            # GnssLogger wraps sentences as  'NMEA,$GNGGA,...*46,<uptimeMillis>' —
            # take the real sentence from the '$' onward, ignoring any wrapper.
            dollar = raw.find("$")
            if dollar < 0:
                continue
            sentence = raw[dollar:].strip()
            if not _checksum_ok(sentence):
                continue
            fields = sentence.split("*")[0].split(",")
            talker = fields[0][-3:]
            if talker == "GGA":
                fix = _parse_gga(fields)
            elif talker == "RMC":
                fix = _parse_rmc(fields)
            else:
                continue
            if fix is None or (fix.lat == 0.0 and fix.lon == 0.0):
                continue

            existing = by_tod.get(fix.tod)
            if existing is None:
                by_tod[fix.tod] = fix
            elif fix.alt is not None and existing.alt is None:
                fix.speed = fix.speed or existing.speed
                by_tod[fix.tod] = fix
            elif fix.speed is not None and existing.speed is None:
                existing.speed = fix.speed

    return [by_tod[t] for t in sorted(by_tod)]
