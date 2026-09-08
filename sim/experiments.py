"""Methods under comparison, and the sweeps that answer the two pilot questions.

Q1  How accurate must the local comparator be for this to work at all?
Q2  Does global inference actually beat greedy composition? (proposal 8.1)
"""
from __future__ import annotations

import numpy as np

from .core import (
    K, N_TRAJ, R, T, REL, forward_backward, log_posterior, marginals_from_posterior,
    onset, query_gts, query_icr, query_tel, shape,
)
from .simulate import (
    NOT_OBSERVED, Config, log_unary_from_obs, sample_timeline, simulate_comparator,
    simulate_unary,
)

RHO_LOW = 0.25          # gate applied to an interval judged unreliable
UNIFORM_LOG_PI = np.log(np.full(K, 1.0 / K))

METHODS = [
    "unary_only",
    "greedy",
    "pairwise_only",
    "no_ctmc",
    "no_gate",
    "naive_missing",
    "full",
]


def _log_ctmc(Ps) -> np.ndarray:
    return np.log(np.clip(np.stack(Ps), 1e-12, None))


def _uniform_log_ctmc() -> np.ndarray:
    return np.full((T - 1, K, K), np.log(1.0 / K))


def greedy_chain(y, p_cmp, Ps, cfg):
    """Argmax relations chained forward from the first per-image read.

    This is the baseline that matters. If it already captures most of the gain,
    the paper's thesis changes (proposal 10). It also exposes how often a chain of
    locally-plausible relations is jointly unrealisable.
    """
    s = np.empty(T, dtype=int)
    s[0] = int(np.argmax(cfg.pi)) if y[0] == NOT_OBSERVED else int(y[0])
    inconsistent = 0
    for t in range(T - 1):
        r = int(np.argmax(p_cmp[t]))
        valid = np.where(REL[s[t]] == r)[0]
        if valid.size == 0:
            inconsistent += 1
            s[t + 1] = int(np.argmax(Ps[t][s[t]]))          # fall back to the prior
        else:
            s[t + 1] = int(valid[np.argmax(Ps[t][s[t], valid])])
    return s, inconsistent


def _point_predictions(traj):
    return int(onset(traj[None, :])[0]), int(shape(traj[None, :])[0])


def run_trial(cfg: Config, rng, icr_pair):
    s, dts, Ps = sample_timeline(cfg, rng)
    p_cmp, reliable = simulate_comparator(s, cfg, rng)
    y = simulate_unary(s, cfg, rng)

    log_cmp = np.log(np.clip(p_cmp, 1e-12, None))
    log_ctmc = _log_ctmc(Ps)
    log_pi = np.log(np.array(cfg.pi))
    gate = np.where(reliable, 1.0, RHO_LOW)
    ones = np.ones(T - 1)

    lu_aware = log_unary_from_obs(y, cfg, missingness_aware=True)
    lu_naive = log_unary_from_obs(y, cfg, missingness_aware=False)
    lu_flat = np.zeros((T, K))

    truth = {
        "tel": int(onset(s[None, :])[0]),
        "gts": int(shape(s[None, :])[0]),
        "icr": int(REL[s[icr_pair[0]], s[icr_pair[1]]]),
    }

    variants = {
        "full":          (lu_aware, log_cmp, log_ctmc, gate),
        "no_gate":       (lu_aware, log_cmp, log_ctmc, ones),
        "no_ctmc":       (lu_aware, log_cmp, _uniform_log_ctmc(), gate),
        "pairwise_only": (lu_flat,  log_cmp, log_ctmc, gate),
        "naive_missing": (lu_naive, log_cmp, log_ctmc, gate),
    }

    out, consistency = {}, {}
    for name, (lu, lc, lg, g) in variants.items():
        post = log_posterior(lu, lc, lg, g, log_pi)
        out[name] = {
            "tel": int(np.argmax(query_tel(post))),
            "gts": int(np.argmax(query_gts(post))),
            "icr": int(np.argmax(query_icr(post, *icr_pair))),
        }
        consistency[name] = 1.0     # a posterior only ever puts mass on valid paths

    # unary-only: independent per-visit argmax, no comparator
    s_un = np.array([
        int(np.argmax(cfg.pi)) if y[t] == NOT_OBSERVED else int(y[t]) for t in range(T)
    ])
    tel_u, gts_u = _point_predictions(s_un)
    out["unary_only"] = {"tel": tel_u, "gts": gts_u,
                         "icr": int(REL[s_un[icr_pair[0]], s_un[icr_pair[1]]])}
    consistency["unary_only"] = 1.0

    s_g, bad = greedy_chain(y, p_cmp, Ps, cfg)
    tel_g, gts_g = _point_predictions(s_g)
    out["greedy"] = {"tel": tel_g, "gts": gts_g,
                     "icr": int(REL[s_g[icr_pair[0]], s_g[icr_pair[1]]])}
    consistency["greedy"] = 0.0 if bad else 1.0

    return truth, out, consistency


def evaluate(cfg: Config, n_trials: int = 4000, seed: int | None = None):
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    hits = {m: {q: 0 for q in ("tel", "gts", "icr")} for m in METHODS}
    cons = {m: 0 for m in METHODS}
    majority = {q: {} for q in ("tel", "gts", "icr")}

    for _ in range(n_trials):
        i = rng.integers(0, T - 1)
        j = rng.integers(i + 1, T)
        truth, preds, consistency = run_trial(cfg, rng, (int(i), int(j)))
        for q in ("tel", "gts", "icr"):
            majority[q][truth[q]] = majority[q].get(truth[q], 0) + 1
        for m in METHODS:
            for q in ("tel", "gts", "icr"):
                hits[m][q] += int(preds[m][q] == truth[q])
            cons[m] += consistency[m]

    res = {}
    for m in METHODS:
        acc = {q: hits[m][q] / n_trials for q in ("tel", "gts", "icr")}
        acc["mean"] = float(np.mean([acc[q] for q in ("tel", "gts", "icr")]))
        acc["consistency"] = cons[m] / n_trials
        res[m] = acc
    res["_majority"] = {
        q: max(majority[q].values()) / n_trials for q in ("tel", "gts", "icr")
    }
    res["_majority"]["mean"] = float(np.mean(list(res["_majority"].values())))
    return res


def sweep_accuracy(levels, n_trials=2500, **overrides):
    rows = []
    for a in levels:
        cfg = Config(cmp_accuracy=a, **overrides)
        r = evaluate(cfg, n_trials=n_trials)
        rows.append((a, r))
    return rows


def sweep_calibration(temps, n_trials=2500, **overrides):
    rows = []
    for temp in temps:
        cfg = Config(cmp_temperature=temp, **overrides)
        r = evaluate(cfg, n_trials=n_trials)
        rows.append((temp, r))
    return rows
