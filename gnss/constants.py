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

# GNSS carrier frequencies [Hz] (only what we need for GPS L1/L5 first).
FREQ_L1 = 1_575.42e6
FREQ_L5 = 1_176.45e6

# Constellation identifiers used in RINEX 3/4 (first char of a satellite id).
SYS_GPS = "G"
SYS_GLONASS = "R"
SYS_GALILEO = "E"
SYS_BEIDOU = "C"
SYS_QZSS = "J"
SYS_SBAS = "S"
