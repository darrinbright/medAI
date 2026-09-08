"""Model misspecification study.

The pilot generated data from the same CTMC family used for inference, so the CTMC prior
was favoured by construction. Here the ground truth comes from processes that each break a
*different* CTMC assumption, the generator Q is then fitted by MLE to that non-CTMC data
(which is what estimating Q_f from Chest ImaGenome would actually be), and inference runs
with the fitted - hence wrong - prior.

Processes and the assumption each violates:
  ctmc           none (control)
  semi_markov    memorylessness  - hazard rises with time already spent in the state
  momentum       first-order Markov - a worsening trend tends to continue
  nonstationary  time-homogeneity - treatment starts mid-timeline and flips the rates
  mixture        homogeneity across patients - fast and slow progressors share one Q
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .core import K, T, birth_death_generator, transition_matrix
from .simulate import Config

H = 1.0  # simulation grid step, days

PROCESSES = ["ctmc", "semi_markov", "momentum", "nonstationary", "mixture"]

PROCESS_VIOLATION = {
    "ctmc": "none (control)",
    "semi_markov": "memorylessness",
    "momentum": "first-order Markov",
    "nonstationary": "time-homogeneity",
    "mixture": "patient homogeneity",
}


def simulate_paths(process: str, n: int, cfg: Config, rng):
    """Vectorised continuous-time simulation on a fine grid, read out at visit times.

    Returns (states (n, T), dts (n, T-1)).
    """
    dts = np.exp(rng.uniform(np.log(cfg.dt_min), np.log(cfg.dt_max), size=(n, T - 1)))
    visit_t = np.concatenate([np.zeros((n, 1)), np.cumsum(dts, axis=1)], axis=1)
    horizon = float(visit_t[:, -1].max())
    grid = np.arange(0.0, horizon + H, H)
    n_steps = grid.size

    up = np.array(cfg.up)      # 0->1, 1->2, 2->3
    down = np.array(cfg.down)  # 1->0, 2->1, 3->2

    s = rng.choice(K, size=n, p=np.array(cfg.pi))
    dwell = np.zeros(n)
    last_dir = np.zeros(n, dtype=int)
    speed = np.where(rng.random(n) < 0.5, 2.5, 0.4) if process == "mixture" else np.ones(n)
    t_tx = rng.uniform(0.15, 0.7, size=n) * horizon   # treatment start, per patient

    S = np.empty((n, n_steps), dtype=np.int8)
    idx = np.arange(n)

    for k in range(n_steps):
        S[:, k] = s
        h_up = np.where(s < K - 1, up[np.clip(s, 0, K - 2)], 0.0)
        h_dn = np.where(s > 0, down[np.clip(s - 1, 0, K - 2)], 0.0)

        if process == "semi_markov":
            m = 1.0 + dwell / 90.0            # increasing hazard: not memoryless
            h_up, h_dn = h_up * m, h_dn * m
        elif process == "momentum":
            m = 2.2
            h_up = h_up * np.where(last_dir > 0, m, np.where(last_dir < 0, 1 / m, 1.0))
            h_dn = h_dn * np.where(last_dir < 0, m, np.where(last_dir > 0, 1 / m, 1.0))
        elif process == "nonstationary":
            pre = grid[k] < t_tx
            h_up = h_up * np.where(pre, 1.8, 0.45)
            h_dn = h_dn * np.where(pre, 0.55, 2.0)
        elif process == "mixture":
            h_up, h_dn = h_up * speed, h_dn * speed

        p_up = np.clip(h_up * H, 0, 0.9)
        p_dn = np.clip(h_dn * H, 0, 0.9)
        u = rng.random(n)
        go_up = u < p_up
        go_dn = (~go_up) & (u < p_up + p_dn)

        moved = go_up | go_dn
        s = s + go_up.astype(int) - go_dn.astype(int)
        s = np.clip(s, 0, K - 1)
        last_dir = np.where(go_up, 1, np.where(go_dn, -1, last_dir))
        dwell = np.where(moved, 0.0, dwell + H)

    pos = np.searchsorted(grid, visit_t, side="right") - 1
    pos = np.clip(pos, 0, n_steps - 1)
    states = S[idx[:, None], pos].astype(int)
    return states, dts


# ------------------------------------------------------------------ CTMC estimation


def _binned_counts(states, dts, n_bins=48):
    """Aggregate (s, s', dt) transitions into log-spaced dt bins.

    Standard practice for CTMC estimation from panel data, and it makes the likelihood
    cheap: one matrix exponential per bin instead of one per transition.
    """
    s0 = states[:, :-1].ravel()
    s1 = states[:, 1:].ravel()
    dt = dts.ravel()
    edges = np.geomspace(dt.min() * 0.999, dt.max() * 1.001, n_bins + 1)
    b = np.clip(np.digitize(dt, edges) - 1, 0, n_bins - 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    N = np.zeros((n_bins, K, K))
    np.add.at(N, (b, s0, s1), 1.0)
    keep = N.sum(axis=(1, 2)) > 0
    return N[keep], centers[keep]


def fit_ctmc_mle(states, dts, n_bins=48):
    """MLE of the six birth-death rates from discretely observed transitions.

    Directly exercises the open question in proposal 3.7: is Q_f estimable from
    longitudinal annotations when the underlying process is not actually a CTMC?
    """
    N, centers = _binned_counts(states, dts, n_bins)

    def nll(theta):
        rates = np.exp(theta)
        Q = birth_death_generator(rates[:3], rates[3:])
        total = 0.0
        for i, dt in enumerate(centers):
            logP = np.log(transition_matrix(Q, float(dt)))
            total -= float((N[i] * logP).sum())
        return total

    best, best_val = None, np.inf
    for start in (-5.0, -4.0, -6.0):
        res = minimize(nll, np.full(6, start), method="L-BFGS-B",
                       bounds=[(-12.0, 1.0)] * 6)
        if res.fun < best_val:
            best, best_val = res.x, res.fun
    rates = np.exp(best)
    return birth_death_generator(rates[:3], rates[3:]), rates
