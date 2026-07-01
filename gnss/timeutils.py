"""Time conversions between calendar/UTC and the GPS time scale.

The RINEX epoch header gives a calendar time already expressed in the GPS
time scale (see the "TIME OF FIRST OBS ... GPS" line).  Satellite ephemeris
maths need that instant as a *GPS week number* plus *seconds of week* (TOW),
while the output KML/CSV want plain UTC.  This module is the single place
that knows how those representations relate.
"""

from __future__ import annotations

import datetime as dt

from .constants import GPS_UTC_LEAP_SECONDS, SECONDS_IN_WEEK

# GPS time started at this instant (in UTC). GPS has no leap seconds, so the
# offset between GPS time and UTC grows by 1 s every time a leap second is
# added to UTC.
GPS_EPOCH = dt.datetime(1980, 1, 6, 0, 0, 0, tzinfo=dt.timezone.utc)


def datetime_to_gps_week_tow(t: dt.datetime) -> tuple[int, float]:
    """Convert a datetime *in the GPS time scale* to (week, seconds_of_week).

    ``t`` must be timezone-aware and already on the GPS scale (i.e. the value
    printed in a RINEX epoch line whose system flag is GPS).  We deliberately
    do NOT apply leap seconds here — the epoch line is not UTC.
    """
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    delta = t - GPS_EPOCH
    total_seconds = delta.total_seconds()
    week = int(total_seconds // SECONDS_IN_WEEK)
    tow = total_seconds - week * SECONDS_IN_WEEK
    return week, tow


def gps_to_utc(t_gps: dt.datetime) -> dt.datetime:
    """Convert a GPS-scale datetime to UTC by removing leap seconds."""
    return t_gps - dt.timedelta(seconds=GPS_UTC_LEAP_SECONDS)


def gps_week_tow_to_utc(week: int, tow: float) -> dt.datetime:
    """Convert (GPS week, TOW) to a UTC datetime."""
    t_gps = GPS_EPOCH + dt.timedelta(seconds=week * SECONDS_IN_WEEK + tow)
    return gps_to_utc(t_gps)


def utc_iso(t: dt.datetime) -> str:
    """Format a UTC datetime as an ISO-8601 string with millisecond precision."""
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
