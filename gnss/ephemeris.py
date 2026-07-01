"""GPS broadcast ephemeris -> satellite ECEF position, velocity and clock.

This is the mathematical heart of GNSS positioning.  Given the Keplerian
orbital elements broadcast by a GPS satellite (a RINEX navigation record), we
reconstruct where that satellite physically was at the instant it transmitted
the signal, and how fast it was moving.  The algorithm follows the "User
Algorithm for Ephemeris Determination" table in IS-GPS-200.

Everything here is GPS-specific for now; Galileo and BeiDou use the same
Keplerian formulation with different constants, so this module extends cleanly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import F_REL, GM, OMEGA_E_DOT, SECONDS_IN_WEEK


@dataclass
class GpsEphemeris:
    """One broadcast ephemeris set for a single GPS satellite."""

    sat: str            # e.g. 'G08'
    gps_week: int       # full GPS week of toe
    toc: float          # clock reference time [s of week]
    toe: float          # ephemeris reference time [s of week]

    # clock polynomial
    af0: float
    af1: float
    af2: float

    # orbital elements
    sqrt_a: float
    e: float
    m0: float
    delta_n: float
    omega0: float       # longitude of ascending node at weekly epoch
    i0: float
    omega: float        # argument of perigee
    omega_dot: float
    idot: float

    # harmonic correction terms
    cuc: float
    cus: float
    crc: float
    crs: float
    cic: float
    cis: float

    tgd: float = 0.0    # group delay (single-frequency correction)
    health: float = 0.0


def _time_from_toe(t_sow: float, toe: float) -> float:
    """Time difference t - toe, corrected for a week rollover."""
    dt = t_sow - toe
    if dt > SECONDS_IN_WEEK / 2:
        dt -= SECONDS_IN_WEEK
    elif dt < -SECONDS_IN_WEEK / 2:
        dt += SECONDS_IN_WEEK
    return dt


def _solve_kepler(mk: float, e: float, tol: float = 1e-12) -> float:
    """Solve Kepler's equation  Mk = Ek - e*sin(Ek)  for eccentric anomaly Ek."""
    ek = mk
    for _ in range(30):
        delta = (ek - e * math.sin(ek) - mk) / (1.0 - e * math.cos(ek))
        ek -= delta
        if abs(delta) < tol:
            break
    return ek


def sat_clock_correction(eph: GpsEphemeris, t_sow: float, ek: float) -> float:
    """Satellite clock bias [s] incl. relativistic term (TGD applied by caller).

    Note ``ek`` (eccentric anomaly) is needed for the relativistic correction,
    so this is called after the orbit is propagated.
    """
    dt = _time_from_toe(t_sow, eph.toc)
    dtr = F_REL * eph.e * eph.sqrt_a * math.sin(ek)   # relativistic eccentricity term
    return eph.af0 + eph.af1 * dt + eph.af2 * dt * dt + dtr


def sat_pos_vel_clock(eph: GpsEphemeris, t_transmit_sow: float):
    """Propagate the orbit to transmit time.

    Returns ``(pos, vel, clock_bias, ek)`` where pos/vel are 3-element ECEF
    tuples in metres and metres/second, clock_bias is in seconds (relativity
    included, TGD NOT yet applied), and ek is the eccentric anomaly.
    """
    a = eph.sqrt_a ** 2
    n0 = math.sqrt(GM / a ** 3)                     # computed mean motion
    tk = _time_from_toe(t_transmit_sow, eph.toe)    # time from ephemeris epoch

    n = n0 + eph.delta_n                            # corrected mean motion
    mk = eph.m0 + n * tk                            # mean anomaly
    ek = _solve_kepler(mk, eph.e)                   # eccentric anomaly

    sin_ek, cos_ek = math.sin(ek), math.cos(ek)
    # true anomaly
    vk = math.atan2(math.sqrt(1.0 - eph.e ** 2) * sin_ek, cos_ek - eph.e)
    phik = vk + eph.omega                           # argument of latitude

    sin_2phi, cos_2phi = math.sin(2 * phik), math.cos(2 * phik)
    # second-harmonic perturbation corrections
    du = eph.cus * sin_2phi + eph.cuc * cos_2phi
    dr = eph.crs * sin_2phi + eph.crc * cos_2phi
    di = eph.cis * sin_2phi + eph.cic * cos_2phi

    uk = phik + du                                  # corrected argument of latitude
    rk = a * (1.0 - eph.e * cos_ek) + dr            # corrected radius
    ik = eph.i0 + di + eph.idot * tk                # corrected inclination

    # position in the orbital plane
    xk_orb = rk * math.cos(uk)
    yk_orb = rk * math.sin(uk)

    # corrected longitude of ascending node (accounts for Earth rotation)
    omega_k = eph.omega0 + (eph.omega_dot - OMEGA_E_DOT) * tk - OMEGA_E_DOT * eph.toe

    sin_ok, cos_ok = math.sin(omega_k), math.cos(omega_k)
    sin_ik, cos_ik = math.sin(ik), math.cos(ik)

    x = xk_orb * cos_ok - yk_orb * cos_ik * sin_ok
    y = xk_orb * sin_ok + yk_orb * cos_ik * cos_ok
    z = yk_orb * sin_ik

    # ---- velocity: analytic derivatives of the above ----
    ek_dot = n / (1.0 - eph.e * cos_ek)
    vk_dot = ek_dot * math.sqrt(1.0 - eph.e ** 2) / (1.0 - eph.e * cos_ek)
    phik_dot = vk_dot

    du_dot = 2 * phik_dot * (eph.cus * cos_2phi - eph.cuc * sin_2phi)
    dr_dot = 2 * phik_dot * (eph.crs * cos_2phi - eph.crc * sin_2phi)
    di_dot = eph.idot + 2 * phik_dot * (eph.cis * cos_2phi - eph.cic * sin_2phi)

    uk_dot = phik_dot + du_dot
    rk_dot = a * eph.e * sin_ek * ek_dot + dr_dot
    ik_dot = di_dot
    omega_k_dot = eph.omega_dot - OMEGA_E_DOT

    xk_orb_dot = rk_dot * math.cos(uk) - rk * uk_dot * math.sin(uk)
    yk_orb_dot = rk_dot * math.sin(uk) + rk * uk_dot * math.cos(uk)

    vx = (
        xk_orb_dot * cos_ok
        - yk_orb_dot * cos_ik * sin_ok
        + yk_orb * sin_ik * sin_ok * ik_dot
        - y * omega_k_dot
    )
    vy = (
        xk_orb_dot * sin_ok
        + yk_orb_dot * cos_ik * cos_ok
        - yk_orb * sin_ik * cos_ok * ik_dot
        + x * omega_k_dot
    )
    vz = yk_orb_dot * sin_ik + yk_orb * cos_ik * ik_dot

    clock = sat_clock_correction(eph, t_transmit_sow, ek)
    return (x, y, z), (vx, vy, vz), clock, ek


def select_ephemeris(ephemerides: list[GpsEphemeris], t_sow: float) -> GpsEphemeris | None:
    """Pick the ephemeris whose reference time (toe) is closest to ``t_sow``."""
    if not ephemerides:
        return None
    return min(ephemerides, key=lambda e: abs(_time_from_toe(t_sow, e.toe)))
