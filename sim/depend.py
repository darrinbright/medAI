"""Conditional independence study (proposal 3.9).

The model factorises as

    P(s_1:T | y, z) prop P(s_1) prod_t P(y_t | s_t) prod_t psi_t(s_t, s_t+1)

which assumes y_t, z_{t-1,t} and z_{t,t+1} are conditionally independent given s_t. They are
not: all three are computed from the same image x_t.

The violation is simulated at its real source. Each image carries a latent nuisance factor -
rotation, inspiration depth, AP magnification, an overlying line - that shifts its *apparent*
severity by delta_t. The per-image read sees the apparent severity, and so does the comparator
at both intervals touching that image. One bad film therefore produces three correlated errors:
an inflated read at V3, a spurious "worsened" on V2->V3, and a spurious "improved" on V3->V4.

Ground truth for evaluation remains the true state sequence.
"""
from __future__ import annotations

import numpy as np

from .core import K, R, T, relation
from .simulate import (
    NOT_OBSERVED, Config, _cmp_dist, calibrate_boost, unary_confusion,
)


def sample_nuisance(cfg: Config, strength: float, rng) -> np.ndarray:
    """delta_t in {-1, 0, +1}: how much image t misrepresents true severity."""
    if strength <= 0:
        return np.zeros(T, dtype=int)
    p_shift = strength / 2.0
    u = rng.random(T)
    return np.where(u < p_shift, -1, np.where(u < 2 * p_shift, 1, 0)).astype(int)


def apparent_states(s: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return np.clip(s + delta, 0, K - 1)


def simulate_correlated(s, cfg: Config, strength: float, rng):
    """Per-image reads and pairwise comparisons, both driven by apparent severity.

    Returns (y, p_cmp, reliable, s_apparent).
    """
    delta = sample_nuisance(cfg, strength, rng)
    s_app = apparent_states(np.asarray(s), delta)

    # per-image read of the APPARENT severity
    C = unary_confusion(cfg)
    mu = np.array(cfg.missing_rate)
    y = np.empty(T, dtype=int)
    for t in range(T):
        if rng.random() < mu[s[t]]:          # missingness still driven by true severity
            y[t] = NOT_OBSERVED
        else:
            y[t] = rng.choice(K, p=C[s_app[t]])

    # comparator sees the APPARENT relation on both intervals touching each image
    reliable = rng.random(T - 1) >= cfg.unreliable_rate
    b_ok = calibrate_boost(cfg, cfg.cmp_accuracy)
    b_bad = calibrate_boost(cfg, cfg.cmp_accuracy_unreliable)
    p_cmp = np.empty((T - 1, R))
    for t in range(T - 1):
        r_app = relation(int(s_app[t]), int(s_app[t + 1]))
        p = _cmp_dist(r_app, b_ok if reliable[t] else b_bad, cfg, rng)
        if cfg.cmp_temperature != 1.0:
            p = p ** (1.0 / cfg.cmp_temperature)
            p /= p.sum()
        p_cmp[t] = p
    return y, p_cmp, reliable, s_app


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    """ECE plus the signed overconfidence gap (mean confidence minus accuracy)."""
    conf = np.asarray(conf, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece), float(conf.mean() - correct.mean())
