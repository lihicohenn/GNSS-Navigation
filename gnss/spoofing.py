"""Integrity monitoring: flag measurements that look spoofed or interfered.

A meaconing/spoofing attack (or strong multipath/interference) leaves fingerprints
in quantities we already compute per epoch.  None of these is proof on its own,
so we score several independent, cheap indicators and raise a flag only when they
agree.  The detectors are:

* **Residual RAIM** — after the least-squares fit, the post-fit pseudorange
  residuals should be a few metres.  A spoofer that cannot perfectly reproduce
  the true geometry (or that mixes real and fake signals) inflates them.
* **Power uniformity** — genuine signals fade with elevation, so C/N0 correlates
  positively with elevation and spans a wide range.  A spoofer replaying one
  amplifier tends to deliver uniformly *high* C/N0 with little elevation
  dependence.
* **Kinematic consistency** — the step between consecutive position fixes must
  match the Doppler-derived velocity.  An injected position jump breaks this.

The output is a per-epoch diagnostic plus a short summary verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .solver import EpochSolution


@dataclass
class EpochDiagnostic:
    index: int
    residual_rms: float
    cn0_mean: float
    cn0_std: float
    cn0_elev_corr: float          # Pearson r between C/N0 and elevation (nan if n<4)
    kinematic_mismatch: float     # |pos step - velocity*dt|  [m]
    flags: list[str] = field(default_factory=list)

    @property
    def suspicious(self) -> bool:
        return bool(self.flags)


@dataclass
class SpoofingReport:
    epochs: list[EpochDiagnostic]
    thresholds: dict

    @property
    def suspicious_epochs(self) -> int:
        return sum(1 for e in self.epochs if e.suspicious)

    @property
    def flag_counts(self) -> dict:
        counts: dict[str, int] = {}
        for e in self.epochs:
            for f in e.flags:
                counts[f] = counts.get(f, 0) + 1
        return counts


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 4 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def analyze(
    solutions: list[EpochSolution],
    residual_rms_m: float = 25.0,
    cn0_uniform_std: float = 2.0,
    cn0_high_mean: float = 45.0,
    kinematic_m: float = 30.0,
) -> SpoofingReport:
    """Screen a solved trajectory for spoofing/interference fingerprints."""
    thresholds = {
        "residual_rms_m": residual_rms_m,
        "cn0_uniform_std": cn0_uniform_std,
        "cn0_high_mean": cn0_high_mean,
        "kinematic_m": kinematic_m,
    }
    diags: list[EpochDiagnostic] = []

    for i, s in enumerate(solutions):
        residuals = np.array([si.residual_m for si in s.sats])
        cn0 = np.array([si.cn0 for si in s.sats])
        elev = np.array([si.elevation_deg for si in s.sats])

        res_rms = float(np.sqrt(np.mean(residuals**2))) if len(residuals) else 0.0
        cn0_mean = float(np.mean(cn0)) if len(cn0) else 0.0
        cn0_std = float(np.std(cn0)) if len(cn0) else 0.0
        corr = _pearson(cn0, elev)

        # kinematic step vs Doppler velocity (needs the previous epoch)
        mismatch = 0.0
        if i > 0:
            prev = solutions[i - 1]
            dt = (s.time_gps - prev.time_gps).total_seconds()
            if 0 < dt <= 5.0:
                step = np.linalg.norm(s.ecef - prev.ecef)
                expected = np.linalg.norm(prev.vel_ecef) * dt
                mismatch = abs(step - expected)

        flags: list[str] = []
        if res_rms > residual_rms_m:
            flags.append("high-residuals")
        # uniformly high power with no elevation dependence
        weak_corr = np.isnan(corr) or corr < 0.1
        if cn0_mean > cn0_high_mean and cn0_std < cn0_uniform_std and weak_corr:
            flags.append("uniform-power")
        if mismatch > kinematic_m:
            flags.append("kinematic-jump")

        diags.append(EpochDiagnostic(
            index=i, residual_rms=res_rms, cn0_mean=cn0_mean, cn0_std=cn0_std,
            cn0_elev_corr=corr, kinematic_mismatch=mismatch, flags=flags,
        ))

    return SpoofingReport(epochs=diags, thresholds=thresholds)


def format_report(report: SpoofingReport, max_examples: int = 5) -> str:
    """Human-readable integrity summary for the console."""
    n = len(report.epochs)
    if n == 0:
        return "      Spoofing analysis: no epochs to analyse."

    suspicious = report.suspicious_epochs
    lines = [
        f"      Spoofing analysis: {suspicious}/{n} epochs flagged "
        f"({100.0 * suspicious / n:.1f}%)"
    ]
    if report.flag_counts:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(report.flag_counts.items()))
        lines.append(f"        by indicator: {detail}")

    examples = [e for e in report.epochs if e.suspicious][:max_examples]
    for e in examples:
        lines.append(
            f"        epoch {e.index}: {', '.join(e.flags)} "
            f"(res_rms={e.residual_rms:.1f} m, C/N0={e.cn0_mean:.0f}±{e.cn0_std:.0f} dB-Hz, "
            f"jump={e.kinematic_mismatch:.1f} m)"
        )
    if suspicious == 0:
        lines.append("        no spoofing/interference fingerprints detected.")
    return "\n".join(lines)
