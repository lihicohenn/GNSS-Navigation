"""Write the computed trajectory to CSV and KML.

CSV is the machine-readable full-precision record (one row per epoch).  KML is
for visual inspection in Google Earth / Google Maps: a coloured track plus a
handful of labelled placemarks so the path is easy to follow.
"""

from __future__ import annotations

import csv
from xml.sax.saxutils import escape

from . import timeutils
from .solver import EpochSolution

CSV_COLUMNS = [
    "utc_time", "lat_deg", "lon_deg", "alt_m",
    "ecef_x_m", "ecef_y_m", "ecef_z_m",
    "vel_e_ms", "vel_n_ms", "vel_u_ms", "speed_ms",
    "clock_bias_m", "n_sats", "systems", "gdop",
]


def write_csv(path: str, solutions: list[EpochSolution]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for s in solutions:
            writer.writerow([
                timeutils.utc_iso(s.time_utc),
                f"{s.lat:.8f}", f"{s.lon:.8f}", f"{s.alt:.3f}",
                f"{s.ecef[0]:.3f}", f"{s.ecef[1]:.3f}", f"{s.ecef[2]:.3f}",
                f"{s.vel_enu[0]:.3f}", f"{s.vel_enu[1]:.3f}", f"{s.vel_enu[2]:.3f}",
                f"{s.speed:.3f}",
                f"{s.clock_bias_m:.3f}", s.n_sats, s.systems, f"{s.gdop:.2f}",
            ])


def _kml_coords(solutions: list[EpochSolution]) -> str:
    # KML wants lon,lat,alt
    return " ".join(f"{s.lon:.8f},{s.lat:.8f},{s.alt:.2f}" for s in solutions)


def _reference_placemark(reference) -> str:
    """A green LineString for a comparison track of (lat, lon) points."""
    if not reference:
        return ""
    coords = " ".join(f"{lon:.8f},{lat:.8f},0" for lat, lon in reference)
    return f"""    <Placemark>
      <name>NMEA reference</name>
      <styleUrl>#reference</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <altitudeMode>clampToGround</altitudeMode>
        <coordinates>{coords}</coordinates>
      </LineString>
    </Placemark>"""


def write_kml(
    path: str,
    solutions: list[EpochSolution],
    label_every: int = 50,
    reference: list[tuple[float, float]] | None = None,
) -> None:
    """Write a KML track (LineString) with periodic labelled placemarks.

    ``reference`` optionally overlays a second (green) track of (lat, lon)
    points — used to plot the phone's own NMEA fix next to our solution.
    """
    coords = _kml_coords(solutions)

    placemarks = []
    for i, s in enumerate(solutions):
        if i % label_every == 0 or i == len(solutions) - 1:
            when = escape(timeutils.utc_iso(s.time_utc))
            placemarks.append(
                f"""    <Placemark>
      <name>{i}</name>
      <description>{when} | {s.speed:.1f} m/s | {s.n_sats} sats [{s.systems}]</description>
      <Point><coordinates>{s.lon:.8f},{s.lat:.8f},{s.alt:.2f}</coordinates></Point>
    </Placemark>"""
            )

    kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>GNSS trajectory</name>
    <Style id="track">
      <LineStyle><color>ff0000ff</color><width>3</width></LineStyle>
    </Style>
    <Style id="reference">
      <LineStyle><color>ff00ff00</color><width>2</width></LineStyle>
    </Style>
    <Placemark>
      <name>Path</name>
      <styleUrl>#track</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <altitudeMode>clampToGround</altitudeMode>
        <coordinates>{coords}</coordinates>
      </LineString>
    </Placemark>
{_reference_placemark(reference)}
{chr(10).join(placemarks)}
  </Document>
</kml>
"""
    with open(path, "w") as fh:
        fh.write(kml)
