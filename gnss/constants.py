"""Physical and GNSS constants (WGS-84 / IS-GPS-200).

All values are SI units (metres, seconds, radians) unless noted.
Keeping every magic number in one place makes the algorithm modules
readable and easy to audit against the official ICDs.
"""

# --- Universal ---
C = 299_792_458.0                 # speed of light in vacuum [m/s]

# --- WGS-84 ellipsoid (used for ECEF <-> geodetic) ---
WGS84_A = 6_378_137.0             # semi-major axis [m]
WGS84_F = 1.0 / 298.257223563     # flattening
WGS84_B = WGS84_A * (1.0 - WGS84_F)          # semi-minor axis [m]
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)         # first eccentricity squared

# --- Earth / GPS orbital constants (IS-GPS-200) ---
GM = 3.986005e14                  # WGS-84 Earth gravitational constant [m^3/s^2]
OMEGA_E_DOT = 7.2921151467e-5     # WGS-84 Earth rotation rate [rad/s]
F_REL = -4.442807633e-10          # relativistic clock correction factor [s/sqrt(m)]

# --- Time system ---
# GPS time is ahead of UTC by an integer number of leap seconds.
# 18 s has been in effect since 2017-01-01 and is current through 2026.
GPS_UTC_LEAP_SECONDS = 18
SECONDS_IN_WEEK = 604_800

# Constellation identifiers used in RINEX 3/4 (first char of a satellite id).
SYS_GPS = "G"
SYS_GLONASS = "R"
SYS_GALILEO = "E"
SYS_BEIDOU = "C"
SYS_QZSS = "J"
SYS_SBAS = "S"

# --- Per-constellation orbital constants ---------------------------------
# Each system defines its ephemeris maths against its own realisation of the
# Earth gravitational constant (GM) and rotation rate (omega_e).  GPS and QZSS
# use the WGS-84 values above; Galileo/BeiDou use the GTRF/CGCS2000 values.
# GLONASS is integrated in PZ-90 (see ephemeris.glonass_pos_vel_clock).
GM_BY_SYS = {
    SYS_GPS: 3.986005e14,
    SYS_QZSS: 3.986005e14,
    SYS_GALILEO: 3.986004418e14,
    SYS_BEIDOU: 3.986004418e14,
    SYS_GLONASS: 3.9860044e14,        # PZ-90.02
}
OMEGA_E_BY_SYS = {
    SYS_GPS: 7.2921151467e-5,
    SYS_QZSS: 7.2921151467e-5,
    SYS_GALILEO: 7.2921151467e-5,
    SYS_BEIDOU: 7.292115e-5,
    SYS_GLONASS: 7.292115e-5,
}

# GLONASS second zonal harmonic and PZ-90 semi-major axis (for orbit integration).
GLO_J2 = 1.0826257e-3
GLO_AE = 6_378_136.0

# --- Inter-system time offsets ------------------------------------------
# The observation epoch is expressed in the GPS time scale.  Galileo (GST) and
# QZSS are aligned with GPS time to within nanoseconds, so their ephemeris
# seconds-of-week can be compared directly.  BeiDou time (BDT) started at
# 2006-01-01 and is a constant 14 s *behind* GPS time, so a satellite's toe (in
# BDT) must be compared against (gps_tow - 14).
BDT_GPST_OFFSET = 14.0

# --- GNSS carrier frequencies [Hz] --------------------------------------
# Indexed by RINEX band digit (the second character of an observation code,
# e.g. the '1' in 'C1C').  Used to pick the Doppler wavelength and to scale the
# ionospheric delay, which is dispersive (~1/f^2).  GLONASS L1/L2 are FDMA and
# depend on the per-satellite channel number k (see carrier_frequency).
FREQ_L1 = 1_575.42e6              # GPS L1 / Galileo E1 / BeiDou B1C  (legacy alias)
FREQ_L5 = 1_176.45e6             # GPS L5 / Galileo E5a / BeiDou B2a  (legacy alias)

_FREQ_BY_SYS_BAND = {
    SYS_GPS: {"1": 1_575.42e6, "2": 1_227.60e6, "5": 1_176.45e6},
    SYS_QZSS: {"1": 1_575.42e6, "2": 1_227.60e6, "5": 1_176.45e6, "6": 1_278.75e6},
    SYS_GALILEO: {
        "1": 1_575.42e6, "5": 1_176.45e6, "6": 1_278.75e6,
        "7": 1_207.140e6, "8": 1_191.795e6,
    },
    SYS_BEIDOU: {
        "1": 1_575.42e6, "2": 1_561.098e6, "5": 1_176.45e6,
        "6": 1_268.52e6, "7": 1_207.140e6, "8": 1_191.795e6,
    },
}
# GLONASS FDMA nominal centres and per-channel spacing [Hz].
_GLO_FDMA = {"1": (1_602.0e6, 562_500.0), "2": (1_246.0e6, 437_500.0)}
_GLO_L3 = 1_202.025e6            # GLONASS L3 CDMA


def carrier_frequency(system: str, band: str, glonass_k: int = 0) -> float:
    """Carrier frequency [Hz] for a RINEX (system, band) pair.

    ``band`` is the single band digit from an observation code (e.g. '1' from
    'C1C').  ``glonass_k`` is the FDMA channel number (-7..+6), needed only for
    GLONASS L1/L2.  Falls back to GPS L1 for anything unrecognised.
    """
    if system == SYS_GLONASS:
        if band in _GLO_FDMA:
            base, spacing = _GLO_FDMA[band]
            return base + glonass_k * spacing
        if band == "3":
            return _GLO_L3
        return _GLO_FDMA["1"][0]
    return _FREQ_BY_SYS_BAND.get(system, _FREQ_BY_SYS_BAND[SYS_GPS]).get(
        band, FREQ_L1
    )
