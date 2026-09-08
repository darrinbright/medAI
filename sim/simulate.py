"""Synthetic generation of timelines and of the noisy observations a comparator would give.

Two knobs matter and are controlled independently:
  * comparator ACCURACY   - how often argmax p_cmp is the true relation
  * comparator CALIBRATION - how well the reported probabilities match correctness

Accuracy is hit by bisecting a logit boost; calibration is distorted afterwards by a
temperature, which leaves argmax (and therefore accuracy) untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import K, R, T, birth_death_generator, relation, transition_matrix

# Which relation confusions are clinically plausible. Errors are not uniform:
# new/worsened both mean "more disease", resolved/improved both mean "less".
_AFF = np.zeros((R, R))
for _a, _b in [(0, 2), (0, 3), (1, 4), (2, 5), (3, 4), (3, 5), (1, 3), (2, 3)]:
    _AFF[_a, _b] = _AFF[_b, _a] = 1.0


@dataclass
class Config:
    # disease dynamics (per-finding CTMC rates, in units of 1/day)
    up: tuple = (0.006, 0.005, 0.003)
    down: tuple = (0.004, 0.006, 0.007)
    pi: tuple = (0.55, 0.22, 0.15, 0.08)

    # inter-visit intervals, log-uniform in days - deliberately irregular
    dt_min: float = 1.0
    dt_max: float = 365.0

    # comparator
    cmp_accuracy: float = 0.60          # top-1 accuracy on reliable intervals
    cmp_accuracy_unreliable: float = 0.34
    cmp_temperature: float = 1.0        # >1 underconfident, <1 overconfident
    cmp_temperature_unreliable: float | None = None  # if set, used on unreliable intervals
    unreliable_rate: float = 0.25       # fraction of intervals with acquisition mismatch
    cmp_noise: float = 0.85
    cmp_affinity: float = 1.1

    # per-image severity read
    unary_accuracy: float = 0.68
    unary_tau: float = 0.9
    # informative (MNAR) missingness: severe findings are far more likely to be reported
    missing_rate: tuple = (0.55, 0.35, 0.15, 0.05)

    seed: int = 0
    _boost_cache: dict = field(default_factory=dict, repr=False)


def make_generator(cfg: Config) -> np.ndarray:
    return birth_death_generator(np.array(cfg.up), np.array(cfg.down))


def sample_timeline(cfg: Config, rng: np.random.Generator):
    """Ground-truth latent trajectory plus the irregular intervals that produced it."""
    Q = make_generator(cfg)
    dts = np.exp(rng.uniform(np.log(cfg.dt_min), np.log(cfg.dt_max), size=T - 1))
    s = np.empty(T, dtype=int)
    s[0] = rng.choice(K, p=np.array(cfg.pi))
    Ps = [transition_matrix(Q, dt) for dt in dts]
    for t in range(T - 1):
        s[t + 1] = rng.choice(K, p=Ps[t][s[t]])
    return s, dts, Ps


# ------------------------------------------------------------------ comparator


def _cmp_dist(true_rel: int, boost: float, cfg: Config, rng) -> np.ndarray:
    logits = cfg.cmp_affinity * _AFF[true_rel] + rng.normal(0, cfg.cmp_noise, size=R)
    logits[true_rel] += boost
    logits -= logits.max()
    p = np.exp(logits)
    return p / p.sum()


def _empirical_accuracy(boost: float, cfg: Config, n: int = 4000) -> float:
    rng = np.random.default_rng(12345)
    rels = rng.integers(0, R, size=n)
    hit = 0
    for r in rels:
        hit += int(np.argmax(_cmp_dist(r, boost, cfg, rng)) == r)
    return hit / n


def calibrate_boost(cfg: Config, target: float) -> float:
    """Bisect the logit boost so the simulated comparator hits a target top-1 accuracy."""
    key = round(target, 4)
    if key in cfg._boost_cache:
        return cfg._boost_cache[key]
    lo, hi = 0.0, 12.0
    for _ in range(34):
        mid = (lo + hi) / 2
        if _empirical_accuracy(mid, cfg) < target:
            lo = mid
        else:
            hi = mid
    cfg._boost_cache[key] = (lo + hi) / 2
    return cfg._boost_cache[key]


def simulate_comparator(s, cfg: Config, rng):
    """Return (p_cmp (T-1,R), reliable mask (T-1,)).

    Unreliable intervals stand in for acquisition mismatch (AP->PA, rotation,
    shallow inspiration): the comparator is markedly worse there, and the
    reliability gate's job is to notice.
    """
    reliable = rng.random(T - 1) >= cfg.unreliable_rate
    b_ok = calibrate_boost(cfg, cfg.cmp_accuracy)
    b_bad = calibrate_boost(cfg, cfg.cmp_accuracy_unreliable)

    out = np.empty((T - 1, R))
    for t in range(T - 1):
        r = relation(int(s[t]), int(s[t + 1]))
        p = _cmp_dist(r, b_ok if reliable[t] else b_bad, cfg, rng)
        temp = cfg.cmp_temperature
        if not reliable[t] and cfg.cmp_temperature_unreliable is not None:
            temp = cfg.cmp_temperature_unreliable
        if temp != 1.0:                     # distort calibration, preserve argmax
            p = p ** (1.0 / temp)
            p /= p.sum()
        out[t] = p
    return out, reliable


# ------------------------------------------------------------------ per-image reads

NOT_OBSERVED = -1


def unary_confusion(cfg: Config) -> np.ndarray:
    """C[s, k] = P(read severity k | true severity s), adjacent levels confusable."""
    d = np.abs(np.arange(K)[:, None] - np.arange(K)[None, :])
    C = np.exp(-d / cfg.unary_tau)
    np.fill_diagonal(C, 0.0)
    C = C / C.sum(axis=1, keepdims=True) * (1 - cfg.unary_accuracy)
    C[np.arange(K), np.arange(K)] = cfg.unary_accuracy
    return C


def simulate_unary(s, cfg: Config, rng):
    """Observations y_t in {NOT_OBSERVED} u {0..K-1}, with informative missingness."""
    C = unary_confusion(cfg)
    mu = np.array(cfg.missing_rate)
    y = np.empty(T, dtype=int)
    for t in range(T):
        if rng.random() < mu[s[t]]:
            y[t] = NOT_OBSERVED
        else:
            y[t] = rng.choice(K, p=C[s[t]])
    return y


def log_unary_from_obs(y, cfg: Config, missingness_aware: bool) -> np.ndarray:
    """Emission log-likelihoods (proposal 3.4).

    missingness_aware=True   P(NOT_OBSERVED | s) = mu(s)      -- MNAR, correct
    missingness_aware=False  "not mentioned" collapsed to "absent" -- the naive bug
    """
    C = unary_confusion(cfg)
    mu = np.array(cfg.missing_rate)
    out = np.empty((T, K))
    for t in range(T):
        if y[t] == NOT_OBSERVED:
            out[t] = mu if missingness_aware else (1 - mu) * C[:, 0]
        else:
            out[t] = (1 - mu) * C[:, y[t]]
    return np.log(np.clip(out, 1e-12, None))
