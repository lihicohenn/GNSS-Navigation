# GNSS-Navigation — RINEX → 1 Hz Path (Ex0)

Compute an **offline navigation path** (3D position + velocity + UTC time, at
1 Hz) from an Android **RINEX 4.0** observation file, and export it as **KML +
CSV**. The solution is computed *from the RINEX measurements only* — the phone's
own NMEA fix is never used (it's only for cross-checking).

This is a from-scratch implementation of **single-point positioning (SPP)**: it
parses the raw pseudoranges and Doppler, reconstructs where each satellite was
using broadcast ephemeris, and solves for the receiver's position, velocity and
clock by weighted least squares.

---

## 1. Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the self-tests (no data needed):
python tests/test_math.py

# Compute a path.  If you have a navigation file, pass it with --nav:
python main.py data/your_obs.rnx --nav data/your_nav.rnx -o output/track

# If you only have the observation file, broadcast ephemeris is downloaded
# automatically for the recording date:
python main.py data/your_obs.rnx -o output/track
```

Outputs: `output/track.csv` (full precision, one row per second) and
`output/track.kml` (open in Google Earth / Google Maps).

---

## 2. Why you need two files

A RINEX **observation** file (`OBSERVATION DATA` in its header) contains, per
second, the **pseudorange**, **Doppler** and **signal strength** to each
satellite — but *not the satellites' positions*. To place a satellite in orbit
you need **ephemeris** (Keplerian orbital elements), which lives in a RINEX
**navigation** file, or in the daily broadcast product (BRDC) you can download
for the recording date. This project accepts either.

> If your shared-folder example only has an observation file, the tool downloads
> the matching broadcast ephemeris automatically (see `gnss/ephemeris_download.py`).

---

## 3. The algorithm (what happens each second)

For every 1 Hz epoch (`>` block in the RINEX file):

1. **Read measurements** — for each GPS satellite: pseudorange `C1C`, Doppler
   `D1C`, carrier-to-noise `S1C`. (`gnss/rinex_obs.py`)

2. **Locate each satellite** — the signal left the satellite at
   `t_tx = t_rx − pseudorange/c`. Propagate the broadcast Keplerian elements to
   `t_tx` to get the satellite ECEF position + velocity, and its clock offset
   (incl. the relativistic and group-delay terms). (`gnss/ephemeris.py`)

3. **Correct the pseudorange** — add back the satellite clock error so the only
   unknowns left are the receiver's. Apply the **Earth-rotation (Sagnac)**
   correction for the ~0.07 s the signal was in flight. (`gnss/solver.py`,
   `gnss/positioning.py`)

4. **Solve position** — 4 unknowns `(x, y, z, receiver_clock_bias)`, one
   pseudorange equation per satellite. The range is non-linear in the unknowns,
   so we linearise and iterate (Gauss–Newton), weighting each satellite by its
   C/N0. Needs ≥ 4 satellites. One round of gross-outlier rejection follows.
   (`gnss/positioning.py::least_squares_position`)

5. **Solve velocity** — the Doppler measurements give range-rate; with the known
   satellite velocities this is a *linear* least-squares for
   `(vx, vy, vz, clock_drift)`. (`gnss/positioning.py::least_squares_velocity`)

6. **Convert & store** — ECEF → geodetic lat/lon/alt (WGS-84), velocity → local
   East-North-Up, GPS time → UTC. (`gnss/coords.py`, `gnss/timeutils.py`)

7. **Write** the collected epochs to CSV + KML. (`gnss/output.py`)

### The positioning equation

For satellite *i* with ECEF position **sᵢ**, the (corrected) pseudorange is

```
ρᵢ = ‖ sᵢ − r ‖ + c·δt_receiver + noise
```

where **r** is the unknown receiver position and `δt_receiver` its clock bias.
Stacking all satellites and linearising about a current estimate gives
`b = H·Δx`, solved in the weighted least-squares sense
`Δx = (Hᵀ W H)⁻¹ Hᵀ W b`, iterated to convergence.

---

## 4. Project layout

```
gnss/
  constants.py          WGS-84 / IS-GPS-200 constants
  timeutils.py          GPS time <-> UTC, week/second-of-week
  coords.py             ECEF <-> geodetic, ECEF -> ENU
  rinex_obs.py          RINEX 3/4 observation parser
  rinex_nav.py          RINEX 3/4 navigation parser (GPS)
  ephemeris.py          Keplerian elements -> satellite pos/vel/clock  (core math)
  positioning.py        weighted least-squares position & velocity solvers
  solver.py             per-epoch orchestration (ties it all together)
  output.py             CSV + KML writers
  ephemeris_download.py broadcast-ephemeris fallback download
main.py                 command-line entry point
tests/test_math.py      self-tests (coords, orbit, parsing, positioning)
```

---

## 5. Current status & roadmap

- [x] RINEX 3/4 observation + navigation parsing
- [x] GPS satellite position/velocity/clock from broadcast ephemeris
- [x] Weighted least-squares position + Doppler velocity
- [x] CSV + KML export, self-tests
- [ ] Validate against the recording's NMEA track
- [ ] Multi-constellation (Galileo / BeiDou / GLONASS) with inter-system clock biases
- [ ] Ionospheric (Klobuchar) + tropospheric corrections
- [ ] Bonus: detect / analyse spoofed measurements

**Milestone:** GPS-only first (this version), then add the other constellations.

---

## 6. References

- IS-GPS-200 — GPS Interface Specification (satellite position user algorithm).
- RINEX 3.05 / 4.00 format definitions (IGS).
- P. Misra & P. Enge, *Global Positioning System: Signals, Measurements, and
  Performance*.
- RTKLIB, `georinex`, and Stanford `gnss-lib-py` as reference implementations.
