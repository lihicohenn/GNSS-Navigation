"""Atmospheric range corrections: ionosphere (Klobuchar) and troposphere.

A pseudorange is stretched by two media the signal passes through:

* the **ionosphere** — a dispersive plasma whose delay scales as ~1/f^2, so it
  can be modelled from the broadcast Klobuchar coefficients (GPS L1) and scaled
  to any frequency.  Daytime zenith delay is several metres.
* the **troposphere** — the neutral lower atmosphere, non-dispersive (same for
  every frequency), modelled from a standard-atmosphere profile via the
  Saastamoinen formula.  Zenith delay is ~2.3 m, growing as 1/sin(elevation).

Both are subtracted from the measured pseudorange before the least-squares
solve.  They need the satellite's azimuth/elevation, hence an approximate
receiver position — so corrections are applied after a first rough fix.
"""

from __future__ import annotations

import math

import numpy as np

from .constants import C
from .coords import ecef_to_enu_matrix, ecef_to_geodetic


def azimuth_elevation(rcv_ecef: np.ndarray, sat_ecef: np.ndarray) -> tuple[float, float]:
    """Azimuth and elevation [rad] of a satellite from a receiver (ENU frame)."""
    lat, lon, _ = ecef_to_geodetic(*rcv_ecef)
    enu = ecef_to_enu_matrix(lat, lon) @ (np.asarray(sat_ecef) - np.asarray(rcv_ecef))
    east, north, up = enu
    az = math.atan2(east, north)
    el = math.atan2(up, math.hypot(east, north))
    return az, el


def klobuchar_delay(
    rcv_ecef: np.ndarray,
    az: float,
    el: float,
    gps_tow: float,
    alpha: tuple,
    beta: tuple,
) -> float:
    """Klobuchar ionospheric delay on GPS **L1** [metres].

    Follows the IS-GPS-200 single-frequency user algorithm.  Angles are in
    radians; ``gps_tow`` is GPS seconds-of-week.  Scale the result by
    ``(FREQ_L1 / f)**2`` for another carrier frequency ``f``.  Returns 0 if the
    satellite is at/below the horizon.
    """
    if el <= 0.0:
        return 0.0

    lat, lon, _ = ecef_to_geodetic(*rcv_ecef)
    phi_u = math.radians(lat) / math.pi          # geodetic lat  [semicircles]
    lam_u = math.radians(lon) / math.pi          # geodetic lon  [semicircles]
    el_sc = el / math.pi                         # elevation     [semicircles]
    az_rad = az

    # earth-centred angle between receiver and the ionospheric pierce point
    psi = 0.0137 / (el_sc + 0.11) - 0.022

    phi_i = phi_u + psi * math.cos(az_rad)       # pierce-point latitude
    phi_i = max(-0.416, min(0.416, phi_i))
    lam_i = lam_u + psi * math.sin(az_rad) / math.cos(phi_i * math.pi)
    phi_m = phi_i + 0.064 * math.cos((lam_i - 1.617) * math.pi)  # geomagnetic lat

    # local time at the pierce point [s]
    t = 43200.0 * lam_i + gps_tow
    t = t % 86400.0

    amp = sum(alpha[n] * phi_m ** n for n in range(4))
    amp = max(amp, 0.0)
    per = sum(beta[n] * phi_m ** n for n in range(4))
    per = max(per, 72000.0)

    x = 2.0 * math.pi * (t - 50400.0) / per
    slant = 1.0 + 16.0 * (0.53 - el_sc) ** 3     # obliquity factor
    if abs(x) < 1.57:
        delay_s = slant * (5e-9 + amp * (1 - x**2 / 2 + x**4 / 24))
    else:
        delay_s = slant * 5e-9
    return delay_s * C


def saastamoinen_delay(el: float, lat_deg: float, height_m: float,
                       humidity: float = 0.7) -> float:
    """Saastamoinen tropospheric delay [metres] (dry + wet, standard atmosphere).

    ``el`` is elevation [rad]; ``height_m`` the receiver ellipsoidal height.
    Returns 0 at/below the horizon or for absurd heights.
    """
    if el <= 0.0 or height_m < -100.0 or height_m > 1e4:
        return 0.0

    hgt = max(height_m, 0.0)
    pres = 1013.25 * (1.0 - 2.2557e-5 * hgt) ** 5.2568     # pressure [hPa]
    temp = 15.0 - 6.5e-3 * hgt + 273.16                    # temperature [K]
    e = 6.108 * humidity * math.exp((17.15 * temp - 4684.0) / (temp - 38.45))

    lat = math.radians(lat_deg)
    z = math.pi / 2.0 - el                                 # zenith angle
    cos_z = math.cos(z)
    dry = 0.0022768 * pres / (
        1.0 - 0.00266 * math.cos(2.0 * lat) - 0.00028 * hgt / 1e3
    ) / cos_z
    wet = 0.002277 * (1255.0 / temp + 0.05) * e / cos_z
    return dry + wet
