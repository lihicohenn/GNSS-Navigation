"""Coordinate transforms on the WGS-84 ellipsoid.

The positioning solver works entirely in Earth-Centred-Earth-Fixed (ECEF)
Cartesian coordinates.  For human-readable output (and KML) we need geodetic
latitude/longitude/height, and to express velocity in a local East-North-Up
frame we need the ECEF->ENU rotation.
"""

from __future__ import annotations

import numpy as np

from .constants import WGS84_A, WGS84_B, WGS84_E2


def ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float, float]:
    """ECEF (metres) -> (lat_deg, lon_deg, height_m) on WGS-84.

    Uses Bowring's closed-form method, which converges to millimetre level in
    a single pass for terrestrial heights — no iteration required.
    """
    lon = np.arctan2(y, x)

    p = np.hypot(x, y)
    if p < 1e-9:  # at/near the poles
        lat = np.pi / 2 * np.sign(z)
        height = abs(z) - WGS84_B
        return np.degrees(lat), np.degrees(lon), height

    # Bowring's auxiliary quantities
    ep2 = (WGS84_A**2 - WGS84_B**2) / WGS84_B**2  # second eccentricity squared
    theta = np.arctan2(z * WGS84_A, p * WGS84_B)
    lat = np.arctan2(
        z + ep2 * WGS84_B * np.sin(theta) ** 3,
        p - WGS84_E2 * WGS84_A * np.cos(theta) ** 3,
    )
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)  # prime vertical radius
    height = p / np.cos(lat) - n

    return np.degrees(lat), np.degrees(lon), height


def geodetic_to_ecef(lat_deg: float, lon_deg: float, height_m: float) -> np.ndarray:
    """(lat_deg, lon_deg, height_m) on WGS-84 -> ECEF vector (metres)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * np.sin(lat) ** 2)
    x = (n + height_m) * np.cos(lat) * np.cos(lon)
    y = (n + height_m) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + height_m) * np.sin(lat)
    return np.array([x, y, z])


def ecef_to_enu_matrix(lat_deg: float, lon_deg: float) -> np.ndarray:
    """Rotation matrix that maps an ECEF vector to local East-North-Up.

    Apply to a *difference* vector or a velocity vector:
        enu = R @ ecef_vector
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    return np.array(
        [
            [-so, co, 0.0],
            [-sl * co, -sl * so, cl],
            [cl * co, cl * so, sl],
        ]
    )
