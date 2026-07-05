# GNSS-Navigation - RINEX → 1 Hz Path (Ex0)

Compute an **offline navigation path** (3D position + velocity + UTC time, at
1 Hz) from an Android **RINEX 3/4** observation file, and export it as **KML +
CSV**. The solution is computed *from the RINEX measurements only* - the phone's
own NMEA fix is never used for positioning (it's only for cross-checking).

This is a from-scratch implementation of **multi-constellation single-point
positioning (SPP)**: it parses the raw pseudoranges and Doppler, reconstructs
where each satellite was using broadcast ephemeris, applies ionospheric and
tropospheric corrections, and solves for the receiver's position, velocity,
per-system clock offsets and inter-system biases by weighted least squares. It
also validates the result against the phone's NMEA track and screens the
measurements for spoofing/interference.

**Constellations:** GPS, Galileo, BeiDou (incl. GEO), QZSS (Keplerian broadcast
ephemeris) and GLONASS (PZ-90 state-vector integration).

---

## 1. Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run the self-tests (no data needed; need only numpy from requirements.txt):
python tests/test_math.py            # coords, orbit, parsing, positioning
python tests/test_integration.py     # end-to-end GPS pipeline
python tests/test_features.py        # multi-GNSS, corrections, NMEA, spoofing

# Or run everything at once with pytest (optional, needs the dev deps):
pip install -r requirements-dev.txt && python -m pytest tests/ -q

# Compute a path. pass the navigation file with with --nav:
python main.py data/your_obs.rnx --nav data/your_nav.rnx -o output/track

# automatically for the recording date:
python main.py data/your_obs.rnx -o output/track

# Cross-check against the phone's own NMEA fix and overlay it on the KML:
python main.py data/your_obs.rnx --nav data/your_nav.rnx --nmea data/phone.nmea

# Restrict constellations (e.g. GPS-only, or GPS+Galileo):
python main.py data/your_obs.rnx --nav data/your_nav.rnx --systems GE
```

Outputs: `output/track.csv` (full precision, one row per second) and
`output/track.kml` (open in Google Earth / Google Maps).

Useful flags: `--systems` (subset of `GERCJ`), `--no-corrections` (disable the
iono/tropo models + elevation mask), `--elevation-mask DEG`, `--nmea FILE`
(validation), `--no-spoof-check`.

---

## 2. The algorithm (what happens each second)

For every 1 Hz epoch (`>` block in the RINEX file):

1. **Read measurements** - for every satellite of every enabled constellation:
   the pseudorange, Doppler and carrier-to-noise on the system's primary band
   (e.g. GPS `C1C/D1C/S1C`, BeiDou `C2I`, GLONASS `C1C`). (`gnss/rinex_obs.py`,
   `gnss/solver.py::_select_observations`)

2. **Locate each satellite** - the signal left the satellite at
   `t_tx = t_rx − pseudorange/c`. Keplerian systems (GPS/Galileo/BeiDou/QZSS)
   propagate their broadcast elements with the IS-GPS-200 user algorithm and the
   system's own `GM`/`ωₑ` (plus the special final rotation for BeiDou GEOs);
   GLONASS integrates its PZ-90 position/velocity/acceleration state with RK4.
   Each yields ECEF position + velocity and a clock offset (incl. relativistic
   and group-delay terms). (`gnss/ephemeris.py`)

3. **Correct the pseudorange** - add back the satellite clock error, subtract the
   modelled **ionospheric** (Klobuchar, scaled to the carrier frequency) and
   **tropospheric** (Saastamoinen) delays, and apply the **Earth-rotation
   (Sagnac)** correction for the ~0.07 s the signal was in flight.
   (`gnss/atmosphere.py`, `gnss/solver.py`, `gnss/positioning.py`)

4. **Solve position** - `3 + (#systems)` unknowns: `(x, y, z)` plus one receiver
   clock offset per constellation. The range is non-linear in the unknowns, so
   we linearise and iterate (Gauss–Newton), weighting each satellite by its C/N0
   and elevation. A first uncorrected fix seeds the elevation/azimuth the
   atmospheric models need, then we re-solve; one round of gross-outlier
   rejection follows. Needs ≥ 4 satellites (more with several systems).
   (`gnss/positioning.py::least_squares_position`)

5. **Solve velocity** - the Doppler measurements give range-rate; with the known
   satellite velocities this is a *linear* least-squares for
   `(vx, vy, vz, clock_drift)`, one shared receiver clock drift.
   (`gnss/positioning.py::least_squares_velocity`)

6. **Convert & store** - ECEF → geodetic lat/lon/alt (WGS-84), velocity → local
   East-North-Up, GPS time → UTC, plus per-satellite diagnostics.
   (`gnss/coords.py`, `gnss/timeutils.py`)

7. **Validate & screen** - compare each fix to the nearest NMEA fix in time
   (`gnss/validate.py`) and screen the residuals / C-N0 / kinematics for spoofing
   fingerprints (`gnss/spoofing.py`).

8. **Write** - the collected epochs to CSV + KML. (`gnss/output.py`)

### The positioning equation

For satellite *i* (of system *s*) with ECEF position **sᵢ**, the (corrected)
pseudorange is

```
ρᵢ = ‖ sᵢ − r ‖ + c·δt_s + noise
```

where **r** is the unknown receiver position and `δt_s` the receiver clock
offset *for that constellation*. The difference `δt_s − δt_GPS` is the
constellation's **inter-system bias** (ISB), which absorbs the systems'
different time scales and receiver hardware delays. Stacking all satellites and
linearising about a current estimate gives `b = H·Δx`, solved in the weighted
least-squares sense `Δx = (Hᵀ W H)⁻¹ Hᵀ W b`, iterated to convergence.

---

## 3. Project layout

```
gnss/
  constants.py          per-system WGS-84 / IS-GPS-200 constants, frequencies
  timeutils.py          GPS time <-> UTC, week/second-of-week
  coords.py             ECEF <-> geodetic, ECEF -> ENU
  rinex_obs.py          RINEX 3/4 observation parser
  rinex_nav.py          RINEX 3/4 navigation parser (GPS/GAL/BDS/QZSS/GLONASS + iono)
  ephemeris.py          broadcast elements -> satellite pos/vel/clock  (core math)
  atmosphere.py         Klobuchar ionosphere + Saastamoinen troposphere + az/el
  positioning.py        weighted least-squares position (with ISB) & velocity
  solver.py             per-epoch orchestration (ties it all together)
  nmea.py               NMEA-0183 parser (GGA/RMC) for cross-checking
  validate.py           compare our track against the NMEA fix
  spoofing.py           integrity monitoring / spoofing-fingerprint detection
  output.py             CSV + KML writers (with optional NMEA overlay)
  ephemeris_download.py broadcast-ephemeris fallback download
main.py                 command-line entry point
tests/test_math.py         self-tests (coords, orbit, parsing, positioning)
tests/test_integration.py  end-to-end GPS pipeline on synthetic RINEX
tests/test_features.py     multi-GNSS, ISB, corrections, NMEA, spoofing
```

---

### 4. Validation

Every feature has a **self-test** (`tests/`, 19 cases) that forward-models
self-consistent measurements and checks the solver recovers the injected truth
(position to ≤ mm; ISB / iono / tropo / spoofing quantities exactly).

The pipeline is also validated **end-to-end on the real GnssLogger recordings**
in `data/` (Samsung SM-S906E, RINEX 4.01, GPS+GLONASS+Galileo+BeiDou+QZSS). Our
RINEX-only solution is compared against the phone's own NMEA fix:

| recording (1 Hz)            | epochs   | median | mean  | p95   |
|-----------------------------|----------|--------|-------|-------|
| `…17_14_34` (static)        | 116/120  | 3.5 m  | 3.8 m | 7.8 m |
| `…17_17_57`                 | 280/280  | 5.5 m  | 6.3 m | 13 m  |
| `…08_44` (43 min, driving)  | 2296/2590| 8.8 m  | 9.0 m | 13 m  |

That is at the accuracy floor of single-frequency smartphone GNSS. (The
GnssLogger `.txt` additionally contains the broadcast Klobuchar coefficients and
per-satellite `SvPosition`/clock the phone computed — useful cross-references,
but our path is computed only from the RINEX pseudoranges + broadcast ephemeris.)

---

## 5. References

- IS-GPS-200 - GPS Interface Specification (satellite position user algorithm,
  Klobuchar single-frequency ionospheric model).
- Galileo OS-SIS-ICD, BeiDou B1I/B3I ICD (GEO orbit transform), GLONASS ICD
  (PZ-90 equations of motion).
- RINEX 3.05 / 4.00 format definitions (IGS).
- J. Saastamoinen, *Atmospheric correction for the troposphere and stratosphere
  in radio ranging of satellites* (1972).
- P. Misra & P. Enge, *Global Positioning System: Signals, Measurements, and
  Performance*.
- RTKLIB, `georinex`, and Stanford `gnss-lib-py` as reference implementations.
