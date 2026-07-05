"""Broadcast ephemeris -> satellite ECEF position, velocity and clock.

This is the mathematical heart of GNSS positioning.  Given the elements a
satellite broadcasts, we reconstruct where it physically was at the instant it
transmitted the signal, and how fast it was moving.

Two families of ephemeris live here:

* **Keplerian** (GPS, Galileo, BeiDou, QZSS) — orbital elements propagated with
  the "User Algorithm for Ephemeris Determination" of IS-GPS-200.  The only
  per-system differences are the gravitational constant ``GM`` and the Earth
  rotation rate ``omega_e`` (:data:`gnss.constants.GM_BY_SYS` /
  ``OMEGA_E_BY_SYS``), plus a special orbit transform for BeiDou GEO satellites.

* **GLONASS** — the broadcast message is not Keplerian at all: it gives the
  satellite's PZ-90 ECEF position, velocity and (lunisolar) acceleration at a
  reference epoch, which we numerically integrate (RK4) to the transmit time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import (
    BDT_GPST_OFFSET,
    F_REL,
    GLO_AE,
    GLO_J2,
    GM,
    GM_BY_SYS,
    OMEGA_E_BY_SYS,
    OMEGA_E_DOT,
    SECONDS_IN_WEEK,
    SYS_BEIDOU,
    SYS_GLONASS,
)

# BeiDou GEO satellites (BDS-2 C01-C05, BDS-3 C59-C63) use a different final
# orbit rotation than the MEO/IGSO satellites (BeiDou ICD, table 4-1).
_BEIDOU_GEO_PRNS = {1, 2, 3, 4, 5, 59, 60, 61, 62, 63}
_COS_5DEG = math.cos(math.radians(-5.0))
_SIN_5DEG = math.sin(math.radians(-5.0))


@dataclass
class GpsEphemeris:
    """One broadcast Keplerian ephemeris set (GPS / Galileo / BeiDou / QZSS).

    The name is kept for backward compatibility; despite it, an instance
    describes any Keplerian-constellation satellite.  The constellation is taken
    from the first character of :attr:`sat` via :attr:`system`.
    """

    sat: str            # e.g. 'G08', 'E11', 'C21'
    gps_week: int       # full week of toe (system's own week numbering)
    toc: float          # clock reference time [s of week, system scale]
    toe: float          # ephemeris reference time [s of week, system scale]

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

    @property
    def system(self) -> str:
        return self.sat[0]


@dataclass
class GlonassEphemeris:
    """One GLONASS broadcast ephemeris set (PZ-90 ECEF state + accelerations).

    ``toe``/``toc`` are stored as *GPS* seconds-of-week (converted from the
    record's UTC epoch at parse time) so they compare directly against the
    GPS-scale transmit time used everywhere else.
    """

    sat: str            # e.g. 'R07'
    toe: float          # reference time tb [GPS s of week]
    toc: float          # clock reference time [GPS s of week] (== toe)
    tau_n: float        # SV clock bias field (-TauN in the ICD) [s]
    gamma_n: float      # relative frequency bias [-]
    pos: np.ndarray     # (3,) PZ-90 ECEF position at tb [m]
    vel: np.ndarray     # (3,) PZ-90 ECEF velocity at tb [m/s]
    acc: np.ndarray     # (3,) lunisolar acceleration at tb [m/s^2]
    freq_num: int = 0   # FDMA channel number k (-7..+6)
    health: float = 0.0

    @property
    def system(self) -> str:
        return self.sat[0]


def gps_tow_to_sys(system: str, gps_tow: float) -> float:
    """Convert a GPS seconds-of-week to a constellation's own time scale.

    Only BeiDou differs materially: BDT runs a constant 14 s behind GPS time.
    Galileo/QZSS are aligned with GPS to nanoseconds, and GLONASS ephemerides
    are pre-converted to the GPS scale, so both pass through unchanged.
    """
    if system == SYS_BEIDOU:
        return (gps_tow - BDT_GPST_OFFSET) % SECONDS_IN_WEEK
    return gps_tow


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


def _geo_position(eph: GpsEphemeris, tk: float, mu: float, omega_e: float):
    """ECEF position of a BeiDou GEO satellite (special final rotation)."""
    a = eph.sqrt_a ** 2
    n0 = math.sqrt(mu / a ** 3)
    mk = eph.m0 + (n0 + eph.delta_n) * tk
    ek = _solve_kepler(mk, eph.e)

    sin_ek, cos_ek = math.sin(ek), math.cos(ek)
    vk = math.atan2(math.sqrt(1.0 - eph.e ** 2) * sin_ek, cos_ek - eph.e)
    phik = vk + eph.omega
    sin_2phi, cos_2phi = math.sin(2 * phik), math.cos(2 * phik)

    uk = phik + eph.cus * sin_2phi + eph.cuc * cos_2phi
    rk = a * (1.0 - eph.e * cos_ek) + eph.crs * sin_2phi + eph.crc * cos_2phi
    ik = eph.i0 + eph.idot * tk + eph.cis * sin_2phi + eph.cic * cos_2phi

    xk_orb = rk * math.cos(uk)
    yk_orb = rk * math.sin(uk)

    # node WITHOUT the -omega_e*tk term (GEO stays in an inertial-ish frame first)
    omega_k = eph.omega0 + eph.omega_dot * tk - omega_e * eph.toe
    sin_ok, cos_ok = math.sin(omega_k), math.cos(omega_k)
    sin_ik, cos_ik = math.sin(ik), math.cos(ik)

    xg = xk_orb * cos_ok - yk_orb * cos_ik * sin_ok
    yg = xk_orb * sin_ok + yk_orb * cos_ik * cos_ok
    zg = yk_orb * sin_ik

    # rotate by -5 deg about X, then by Earth rotation omega_e*tk about Z
    sino, coso = math.sin(omega_e * tk), math.cos(omega_e * tk)
    x = xg * coso + yg * sino * _COS_5DEG + zg * sino * _SIN_5DEG
    y = -xg * sino + yg * coso * _COS_5DEG + zg * coso * _SIN_5DEG
    z = -yg * _SIN_5DEG + zg * _COS_5DEG
    return np.array([x, y, z]), ek


def _kepler_pos_vel(eph: GpsEphemeris, tk: float, mu: float, omega_e: float):
    """ECEF position + velocity of a MEO/IGSO Keplerian satellite (analytic)."""
    a = eph.sqrt_a ** 2
    n0 = math.sqrt(mu / a ** 3)                     # computed mean motion
    n = n0 + eph.delta_n                            # corrected mean motion
    mk = eph.m0 + n * tk                            # mean anomaly
    ek = _solve_kepler(mk, eph.e)                   # eccentric anomaly

    sin_ek, cos_ek = math.sin(ek), math.cos(ek)
    vk = math.atan2(math.sqrt(1.0 - eph.e ** 2) * sin_ek, cos_ek - eph.e)
    phik = vk + eph.omega                           # argument of latitude

    sin_2phi, cos_2phi = math.sin(2 * phik), math.cos(2 * phik)
    du = eph.cus * sin_2phi + eph.cuc * cos_2phi
    dr = eph.crs * sin_2phi + eph.crc * cos_2phi
    di = eph.cis * sin_2phi + eph.cic * cos_2phi

    uk = phik + du                                  # corrected argument of latitude
    rk = a * (1.0 - eph.e * cos_ek) + dr            # corrected radius
    ik = eph.i0 + di + eph.idot * tk                # corrected inclination

    xk_orb = rk * math.cos(uk)
    yk_orb = rk * math.sin(uk)

    omega_k = eph.omega0 + (eph.omega_dot - omega_e) * tk - omega_e * eph.toe
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
    omega_k_dot = eph.omega_dot - omega_e

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

    return np.array([x, y, z]), np.array([vx, vy, vz]), ek


def sat_pos_vel_clock(eph: GpsEphemeris, t_transmit_sow: float):
    """Propagate a Keplerian orbit to transmit time.

    Returns ``(pos, vel, clock_bias, ek)`` where pos/vel are 3-tuples in ECEF
    metres and metres/second, clock_bias is in seconds (relativity included, TGD
    NOT yet applied), and ek is the eccentric anomaly.  ``t_transmit_sow`` must
    already be in the satellite's own time scale (see :func:`gps_tow_to_sys`).
    """
    sys = eph.system
    mu = GM_BY_SYS.get(sys, GM)
    omega_e = OMEGA_E_BY_SYS.get(sys, OMEGA_E_DOT)
    tk = _time_from_toe(t_transmit_sow, eph.toe)

    prn = int(eph.sat[1:])
    if sys == SYS_BEIDOU and prn in _BEIDOU_GEO_PRNS:
        pos, ek = _geo_position(eph, tk, mu, omega_e)
        dt = 1e-3                                    # finite-difference velocity
        pos2, _ = _geo_position(eph, tk + dt, mu, omega_e)
        vel = (pos2 - pos) / dt
    else:
        pos, vel, ek = _kepler_pos_vel(eph, tk, mu, omega_e)

    clock = sat_clock_correction(eph, t_transmit_sow, ek)
    return tuple(pos), tuple(vel), clock, ek


# --------------------------------------------------------------------------
# GLONASS: numerical integration of the PZ-90 equations of motion
# --------------------------------------------------------------------------

_GLO_STEP = 60.0     # RK4 integration step [s]


def _glonass_deriv(state: np.ndarray, acc: np.ndarray, mu: float,
                   omega_e: float) -> np.ndarray:
    """Time derivative of the GLONASS ECEF state [x,y,z, vx,vy,vz].

    Central gravity + J2 oblateness + centrifugal + Coriolis terms in the
    rotating PZ-90 frame, plus the broadcast lunisolar acceleration ``acc``.
    (Follows the GLONASS ICD / RTKLIB ``deq``.)
    """
    r = state[:3]
    v = state[3:]
    r2 = float(r @ r)
    r3 = r2 * math.sqrt(r2)
    a = 1.5 * GLO_J2 * mu * GLO_AE ** 2 / r2 / r3    # (3/2) J2 mu Ae^2 / r^5
    b = 5.0 * r[2] ** 2 / r2
    c = -mu / r3 - a * (1.0 - b)
    omg2 = omega_e ** 2

    dstate = np.empty(6)
    dstate[:3] = v
    dstate[3] = (c + omg2) * r[0] + 2.0 * omega_e * v[1] + acc[0]
    dstate[4] = (c + omg2) * r[1] - 2.0 * omega_e * v[0] + acc[1]
    dstate[5] = (c - 2.0 * a) * r[2] + acc[2]
    return dstate


def glonass_pos_vel_clock(eph: GlonassEphemeris, t_transmit_sow: float):
    """Integrate a GLONASS orbit from its reference epoch to transmit time.

    Returns ``(pos, vel, clock_bias, None)`` matching the Keplerian signature
    (there is no eccentric anomaly).  ``t_transmit_sow`` is GPS seconds-of-week.
    """
    mu = GM_BY_SYS[SYS_GLONASS]
    omega_e = OMEGA_E_BY_SYS[SYS_GLONASS]
    tk = _time_from_toe(t_transmit_sow, eph.toe)

    state = np.concatenate([eph.pos, eph.vel])
    acc = eph.acc
    t = 0.0
    while abs(tk - t) > 1e-9:
        h = math.copysign(_GLO_STEP, tk - t)
        if abs(tk - t) < abs(h):
            h = tk - t
        k1 = _glonass_deriv(state, acc, mu, omega_e)
        k2 = _glonass_deriv(state + 0.5 * h * k1, acc, mu, omega_e)
        k3 = _glonass_deriv(state + 0.5 * h * k2, acc, mu, omega_e)
        k4 = _glonass_deriv(state + h * k3, acc, mu, omega_e)
        state = state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h

    # ICD clock: dt_sat = -TauN + GammaN*(t-tb); the RINEX field already holds
    # -TauN, so it is added directly (same "pr + c*dt_sat" convention as GPS).
    clock = eph.tau_n + eph.gamma_n * tk
    return tuple(state[:3]), tuple(state[3:]), clock, None


def state_pos_vel_clock(eph, t_transmit_sow: float):
    """Dispatch to the Keplerian or GLONASS propagator based on ephemeris type."""
    if isinstance(eph, GlonassEphemeris):
        return glonass_pos_vel_clock(eph, t_transmit_sow)
    return sat_pos_vel_clock(eph, t_transmit_sow)


def select_ephemeris(ephemerides: list, t_sow: float):
    """Pick the ephemeris whose reference time (toe) is closest to ``t_sow``.

    ``t_sow`` must be in the same time scale as the stored ``toe`` (GPS scale for
    GLONASS, the constellation scale for everything else).
    """
    if not ephemerides:
        return None
    return min(ephemerides, key=lambda e: abs(_time_from_toe(t_sow, e.toe)))
