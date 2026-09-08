"""Latent ordinal trajectory model: states, relation partition, CTMC prior, exact inference.

Everything here is data-independent. The point of this module is to make the model in
docs/proposal.md executable so its behaviour can be measured before any images exist.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.linalg import expm

# ---------------------------------------------------------------- state space

K = 4  # 0 absent, 1 mild, 2 moderate, 3 severe
T = 5  # visits per timeline
STATE_NAMES = ["absent", "mild", "moderate", "severe"]

ABSENT_BOTH, NEW, RESOLVED, STABLE, WORSENED, IMPROVED = range(6)
R = 6
RELATION_NAMES = ["absent-both", "new", "resolved", "stable", "worsened", "improved"]


def relation(s: int, sp: int) -> int:
    """The six clinical change relations. Exactly partitions S x S (see tests)."""
    if s == 0 and sp == 0:
        return ABSENT_BOTH
    if s == 0:
        return NEW
    if sp == 0:
        return RESOLVED
    if sp == s:
        return STABLE
    return WORSENED if sp > s else IMPROVED


REL = np.array([[relation(s, sp) for sp in range(K)] for s in range(K)], dtype=int)

# ---------------------------------------------------------------- CTMC prior


def birth_death_generator(up: np.ndarray, down: np.ndarray) -> np.ndarray:
    """Tridiagonal generator on the ordinal chain 0<->1<->2<->3.

    `up` holds q01,q12,q23 and `down` holds q10,q21,q32 - six free rates per finding,
    as in proposal 3.7. Rows sum to zero.
    """
    Q = np.zeros((K, K))
    for i in range(K - 1):
        Q[i, i + 1] = up[i]
        Q[i + 1, i] = down[i]
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def transition_matrix(Q: np.ndarray, dt: float) -> np.ndarray:
    """P(s' | s, dt) = [exp(Q dt)]. Clipped and renormalised against fp error."""
    P = expm(Q * dt)
    P = np.clip(P, 1e-12, None)
    return P / P.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- trajectories

TRAJ = np.array(list(itertools.product(range(K), repeat=T)), dtype=int)  # (K^T, T)
N_TRAJ = TRAJ.shape[0]


def log_posterior(
    log_unary: np.ndarray,      # (T, K)      log P(y_t | s_t)
    log_pcmp: np.ndarray,       # (T-1, R)    log comparator distribution
    log_ctmc: np.ndarray,       # (T-1, K, K) log P(s_{t+1} | s_t, dtau_t)
    rho: np.ndarray,            # (T-1,)      reliability gate in (0, 1]
    log_pi: np.ndarray,         # (K,)        log P(s_1)
) -> np.ndarray:
    """Exact log posterior over all K^T trajectories, by enumeration.

    Implements proposal 3.3:
        P(s_1:T | y, z) prop P(s_1) prod_t P(y_t|s_t) prod_t psi_t(s_t, s_t+1)
        psi_t(s,s')     prop p_cmp(r(s,s'))^rho_t * P_ctmc(s'|s, dtau_t)

    Exact *in this model* - not a claim about the patient (proposal 3.8).
    """
    score = log_pi[TRAJ[:, 0]].astype(float)
    for t in range(T):
        score += log_unary[t, TRAJ[:, t]]
    for t in range(T - 1):
        s, sp = TRAJ[:, t], TRAJ[:, t + 1]
        score += rho[t] * log_pcmp[t, REL[s, sp]]
        score += log_ctmc[t, s, sp]
    score -= score.max()
    p = np.exp(score)
    return p / p.sum()


def forward_backward(log_unary, log_pcmp, log_ctmc, rho, log_pi) -> np.ndarray:
    """Per-visit marginals via forward-backward. Used to cross-check enumeration."""
    log_A = np.empty((T - 1, K, K))
    for t in range(T - 1):
        log_A[t] = rho[t] * log_pcmp[t][REL] + log_ctmc[t]

    log_a = np.empty((T, K))
    log_a[0] = log_pi + log_unary[0]
    for t in range(1, T):
        m = log_a[t - 1][:, None] + log_A[t - 1]
        log_a[t] = _logsumexp(m, axis=0) + log_unary[t]

    log_b = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        m = log_A[t] + (log_unary[t + 1] + log_b[t + 1])[None, :]
        log_b[t] = _logsumexp(m, axis=1)

    log_g = log_a + log_b
    log_g -= _logsumexp(log_g, axis=1, keepdims=True)
    return np.exp(log_g)


def _logsumexp(x, axis=None, keepdims=False):
    m = np.max(x, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


def marginals_from_posterior(post: np.ndarray) -> np.ndarray:
    """(T, K) per-visit marginals from a trajectory posterior."""
    g = np.zeros((T, K))
    for t in range(T):
        np.add.at(g[t], TRAJ[:, t], post)
    return g


# ---------------------------------------------------------------- query read-outs
#
# All three MI-CXR question families are queries against ONE posterior
# (proposal 3.10), which is what makes their answers mutually consistent.

N_ONSET = T + 1  # never, or first-present visit index 1..T


def onset(traj: np.ndarray) -> np.ndarray:
    """TEL: index of first visit with s>0; 0 means never present."""
    present = traj > 0
    any_present = present.any(axis=-1)
    first = np.argmax(present, axis=-1) + 1
    return np.where(any_present, first, 0)


N_SHAPE = 6
SHAPE_NAMES = [
    "never", "resolved-by-end", "net-worse", "net-better", "stable", "fluctuating",
]


def shape(traj: np.ndarray) -> np.ndarray:
    """GTS: overall course. Six mutually exclusive, exhaustive classes.

    "fluctuating" is a real clinical course, not a leftover bucket: a finding that
    appears, worsens and then improves back to its onset severity - e.g. [0,0,1,2,1] -
    has equal first and last severity but is not stable, and calling it stable would
    discard the peak a radiologist would report.
    """
    traj = np.atleast_2d(traj)
    n, width = traj.shape
    present = traj > 0
    any_present = present.any(axis=1)
    first_idx = np.argmax(present, axis=1)
    first_sev = traj[np.arange(n), first_idx]
    last = traj[:, -1]
    post_onset = np.arange(width)[None, :] >= first_idx[:, None]
    varies = ((traj != first_sev[:, None]) & post_onset).any(axis=1)

    out = np.full(n, 5)                                    # fluctuating
    out[(last == first_sev) & ~varies] = 4                 # stable
    out[last > first_sev] = 2                              # net-worse
    out[last < first_sev] = 3                              # net-better
    out[any_present & (last == 0)] = 1                     # resolved by end
    out[~any_present] = 0                                  # never
    return out


TRAJ_ONSET = onset(TRAJ)
TRAJ_SHAPE = shape(TRAJ)


def _aggregate(post: np.ndarray, labels: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n)
    np.add.at(out, labels, post)
    return out


def query_tel(post: np.ndarray) -> np.ndarray:
    return _aggregate(post, TRAJ_ONSET, N_ONSET)


def query_gts(post: np.ndarray) -> np.ndarray:
    return _aggregate(post, TRAJ_SHAPE, N_SHAPE)


def query_icr(post: np.ndarray, i: int, j: int) -> np.ndarray:
    """ICR: posterior over the relation between visits i and j (0-indexed)."""
    return _aggregate(post, REL[TRAJ[:, i], TRAJ[:, j]], R)


def is_realisable(traj_seq: np.ndarray) -> bool:
    """Trajectory Consistency: does this sequence correspond to a valid path?

    Every K^T sequence is realisable under the defined model, so this only ever
    fails for methods that emit a chain of *relations* which no state sequence
    satisfies - which is exactly what greedy composition can do.
    """
    return bool(np.all((traj_seq >= 0) & (traj_seq < K)))
