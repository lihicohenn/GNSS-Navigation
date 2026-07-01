"""RINEX -> trajectory GNSS positioning package.

Pipeline overview (see README.md for the full algorithm write-up):

    rinex_obs  ─┐
                ├─►  solver.solve_epoch ──►  output.write_csv / write_kml
    rinex_nav ──┘        (uses ephemeris + positioning + coords + timeutils)
"""

__version__ = "0.1.0"
