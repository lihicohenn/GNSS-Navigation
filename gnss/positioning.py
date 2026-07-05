"""Single-point positioning: least-squares solvers for position and velocity.

Position is a *non-linear* problem (range is a square-root of the unknowns),
so we linearise around a current estimate and iterate (Gauss-Newton).  Velocity
is *linear* once the geometry is known, so a single weighted least-squares pass
suffices.

Both solvers work in ECEF metres.  The unknown vectors are:
    position:  [dx, dy, dz, c*dt_G, c*dt_E, ...]  (3 + one clock per system)
    velocity:  [vx, vy, vz, c*dt_receiver_rate]   (4 unknowns)

Mixing constellations means each system keeps its own receiver-clock offset:
the difference between two of them is that pair's **inter-system bias** (ISB),
which absorbs the fact that the systems' time scales and the receiver's hardware
delays are not identical.  With one system present this reduces to the classic
4-unknown GPS solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import C, OMEGA_E_DOT


def weight_from_cn0(cn0: float) -> float:
    """Measurement weight from carrier-to-noise density (dB-Hz).

    Stronger signals get more weight.  Using linear C/N0 (10^(cn0/10)) is a
    common, simple choice that de-weights weak/multipath-prone satellites.
    """
    return 10.0 ** (cn0 / 10.0)


def earth_rotation_correction(sat_pos: np.ndarray, travel_time: float) -> np.ndarray:
    """Rotate a satellite ECEF position into the receive-time ECEF frame.

    During the signal's flight the Earth (and the ECEF frame) rotates by
    theta = omega_e * travel_time.  This "Sagnac" correction is ~30 m and must
    not be skipped.
    """
    theta = OMEGA_E_DOT * travel_time
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return rot @ sat_pos


@dataclass
class PositionSolution:
    pos: np.ndarray          # ECEF position [m]
    clock_bias: float        # receiver clock bias of the reference system [m]
    clock_biases: np.ndarray # per-system clock biases [m] (one entry per system)
    residuals: np.ndarray    # post-fit residuals [m]
    gdop: float              # geometric dilution of precision
    n_sats: int
    converged: bool


def least_squares_position(
    sat_positions: np.ndarray,      # (n, 3) ECEF at transmit time (Sagnac applied per-iter)
    pseudoranges: np.ndarray,       # (n,) corrected pseudoranges [m]
    weights: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    sys_index: np.ndarray | None = None,   # (n,) clock-column index per satellite
    max_iter: int = 10,
    tol: float = 1e-4,
) -> PositionSolution:
    """Weighted Gauss-Newton solve for receiver position + per-system clock bias.

    ``sys_index`` labels which receiver-clock unknown each satellite belongs to
    (0 = reference system).  With it omitted, every satellite shares one clock
    and the solve is the classic 4-unknown problem.  The number of clock
    unknowns is ``max(sys_index) + 1``; each contributes one column of ones to
    the design matrix restricted to its own satellites.
    """
    n = len(pseudoranges)
    if sys_index is None:
        sys_index = np.zeros(n, dtype=int)
    else:
        sys_index = np.asarray(sys_index, dtype=int)
    n_clock = int(sys_index.max()) + 1 if n else 1
    n_unknown = 3 + n_clock

    x = np.zeros(n_unknown)
    if x0 is not None:
        x0 = np.asarray(x0, dtype=float)
        x[:3] = x0[:3]
        if len(x0) > 3:                       # seed every clock with the prior bias
            x[3:] = x0[3]

    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    W = np.diag(w)

    converged = False
    residuals = np.zeros(n)
    H = np.zeros((n, n_unknown))
    for _ in range(max_iter):
        rcv = x[:3]

        b = np.zeros(n)
        for i in range(n):
            travel_time = np.linalg.norm(sat_positions[i] - rcv) / C
            sp = earth_rotation_correction(sat_positions[i], travel_time)
            diff = sp - rcv
            rng = np.linalg.norm(diff)
            los = diff / rng                      # unit vector receiver -> satellite
            cdt = x[3 + sys_index[i]]
            b[i] = pseudoranges[i] - (rng + cdt)
            H[i, :3] = -los
            H[i, 3 + sys_index[i]] = 1.0

        # normal equations:  (H^T W H) dx = H^T W b
        HtW = H.T @ W
        try:
            dx = np.linalg.inv(HtW @ H) @ HtW @ b
        except np.linalg.LinAlgError:
            break                                 # rank-deficient geometry -> give up
        x = x + dx
        residuals = b
        if np.linalg.norm(dx[:3]) < tol:
            converged = True
            break

    # GDOP from the UNWEIGHTED geometry (the standard dilution of precision:
    # position + reference-system clock), a meaningful quality metric.
    try:
        gcov = np.linalg.inv(H.T @ H)
        gdop = float(np.sqrt(np.trace(gcov[:4, :4])))
    except np.linalg.LinAlgError:
        gdop = float("inf")
    return PositionSolution(
        pos=x[:3], clock_bias=x[3], clock_biases=x[3:].copy(),
        residuals=residuals, gdop=gdop, n_sats=n, converged=converged,
    )


def least_squares_velocity(
    sat_positions: np.ndarray,      # (n, 3) ECEF [m]
    sat_velocities: np.ndarray,     # (n, 3) ECEF [m/s]
    rcv_pos: np.ndarray,            # (3,) receiver ECEF [m]
    range_rates: np.ndarray,        # (n,) observed range rate [m/s] = -lambda*Doppler
    sat_clock_rates: np.ndarray,    # (n,) satellite clock drift [m/s] (c * d(dt_sat)/dt)
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Linear weighted least-squares for receiver velocity + clock drift.

    Returns ``(velocity_ecef, clock_drift)`` (m/s, and m/s for c*dt_rate).
    """
    n = len(range_rates)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    W = np.diag(w)

    H = np.zeros((n, 4))
    b = np.zeros(n)
    for i in range(n):
        diff = sat_positions[i] - rcv_pos
        los = diff / np.linalg.norm(diff)
        # observed range rate minus the part explained by satellite motion/clock
        b[i] = range_rates[i] - los @ sat_velocities[i] + sat_clock_rates[i]
        H[i, :3] = -los
        H[i, 3] = 1.0

    HtW = H.T @ W
    sol = np.linalg.inv(HtW @ H) @ HtW @ b
    return sol[:3], float(sol[3])


def elevation_deg(rcv_pos: np.ndarray, sat_pos: np.ndarray) -> float:
    """Elevation angle [deg] of a satellite as seen from the receiver."""
    up = rcv_pos / np.linalg.norm(rcv_pos)         # geocentric up (good enough)
    diff = sat_pos - rcv_pos
    los = diff / np.linalg.norm(diff)
    return float(np.degrees(np.arcsin(np.clip(los @ up, -1.0, 1.0))))
