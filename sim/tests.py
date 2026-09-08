"""Correctness checks. Run before trusting any sweep output."""
from __future__ import annotations

import itertools

import numpy as np

from .core import (
    K, N_TRAJ, R, REL, T, TRAJ, forward_backward, log_posterior,
    marginals_from_posterior, onset, relation, shape, transition_matrix,
)
from .simulate import Config, make_generator


def test_partition_exact():
    """The six relations tile S x S: 16 cells, no overlap, no remainder."""
    counts = np.bincount(REL.ravel(), minlength=R)
    assert counts.sum() == K * K == 16, counts
    assert list(counts) == [1, 3, 3, 3, 3, 3], counts
    cells = [set() for _ in range(R)]
    for s, sp in itertools.product(range(K), repeat=2):
        cells[relation(s, sp)].add((s, sp))
    assert set().union(*cells) == set(itertools.product(range(K), repeat=2))
    assert sum(len(c) for c in cells) == 16
    return "partition: 16/16 cells, counts [1,3,3,3,3,3]"


def test_generator_rows_sum_zero():
    Q = make_generator(Config())
    assert np.allclose(Q.sum(axis=1), 0), Q.sum(axis=1)
    off = Q[~np.eye(K, dtype=bool)]
    assert (off >= 0).all()
    return "generator: rows sum to 0, off-diagonals non-negative"


def test_ctmc_is_stochastic():
    Q = make_generator(Config())
    for dt in (0.5, 1.0, 14.0, 365.0, 3650.0):
        P = transition_matrix(Q, dt)
        assert np.allclose(P.sum(axis=1), 1), (dt, P.sum(axis=1))
        assert (P >= 0).all()
    return "CTMC: exp(Q dt) row-stochastic across dt in [0.5, 3650] days"


def test_ctmc_semigroup():
    """exp(Q(a+b)) == exp(Qa) exp(Qb) - the property that makes irregular dt valid."""
    Q = make_generator(Config())
    a, b = 7.0, 23.0
    assert np.allclose(transition_matrix(Q, a + b),
                       transition_matrix(Q, a) @ transition_matrix(Q, b), atol=1e-9)
    return "CTMC: semigroup property holds (irregular intervals compose correctly)"


def test_enumeration_matches_forward_backward():
    """Exact inference two ways. If these disagree, nothing downstream is trustworthy."""
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(40):
        lu = np.log(rng.dirichlet(np.ones(K), size=T))
        lc = np.log(rng.dirichlet(np.ones(R), size=T - 1))
        lg = np.log(np.stack([rng.dirichlet(np.ones(K), size=K) for _ in range(T - 1)]))
        rho = rng.uniform(0.1, 1.0, size=T - 1)
        lpi = np.log(rng.dirichlet(np.ones(K)))
        m_enum = marginals_from_posterior(log_posterior(lu, lc, lg, rho, lpi))
        m_fb = forward_backward(lu, lc, lg, rho, lpi)
        worst = max(worst, float(np.abs(m_enum - m_fb).max()))
    assert worst < 1e-9, worst
    return f"inference: enumeration == forward-backward (max abs diff {worst:.2e})"


def test_trajectory_space():
    assert N_TRAJ == K ** T == 1024
    assert TRAJ.shape == (1024, T)
    assert len({tuple(r) for r in TRAJ}) == 1024
    return "trajectory space: 1024 distinct paths enumerated"


def test_query_labels_exhaustive():
    o, sh = onset(TRAJ), shape(TRAJ)
    assert set(o.tolist()) <= set(range(T + 1))
    assert set(sh.tolist()) == set(range(6)), sorted(set(sh.tolist()))
    assert onset(np.zeros((1, T), dtype=int))[0] == 0
    assert onset(np.array([[0, 0, 1, 2, 1]]))[0] == 3
    assert shape(np.array([[0, 0, 0, 0, 0]]))[0] == 0      # never
    assert shape(np.array([[0, 1, 2, 1, 0]]))[0] == 1      # resolved by end
    assert shape(np.array([[0, 1, 1, 2, 3]]))[0] == 2      # net-worse
    assert shape(np.array([[3, 2, 2, 1, 1]]))[0] == 3      # net-better
    assert shape(np.array([[0, 2, 2, 2, 2]]))[0] == 4      # stable after onset
    assert shape(np.array([[0, 0, 1, 2, 1]]))[0] == 5      # peaked -> fluctuating
    return "queries: TEL/GTS label maps exhaustive and correct on spot checks"


def test_finding_polarity():
    """Magnitude verbs invert for quantity-named findings; evaluative verbs never do.

    Exclusivity cannot catch a polarity error - flipping two options consistently keeps them
    distinct - so these are hand-labelled cases drawn from the real corpus.
    """
    from .core import IMPROVED, WORSENED
    from .gts_parse import option_relations

    cases = [
        # (text, interval, expected relation, why)
        ("Between T1 and T2, lung volume decreases from adequate to severely low aeration.",
         1, WORSENED, "magnitude down on an inverted finding"),
        ("Between T1 and T2, lung volumes increase.",
         1, IMPROVED, "magnitude up on an inverted finding"),
        ("Between T2 and T3, expansion of both lungs has decreased.",
         2, WORSENED, "inverted finding, expansion"),
        ("Between T1 and T2, low lung volume improves to adequate aeration.",
         1, IMPROVED, "evaluative verb ignores polarity"),
        ("Between T1 and T2, low lung volume worsens to near-complete collapse.",
         1, WORSENED, "evaluative verb ignores polarity"),
        ("Between T1 and T2, lower lobe volume loss diminishes.",
         1, IMPROVED, "names the abnormality, so NOT inverted"),
        ("Between T1 and T2, hyperinflation increases.",
         1, WORSENED, "names the abnormality, so NOT inverted"),
        ("Between T1 and T2, there is a moderate right pleural effusion that is decreasing.",
         1, IMPROVED, "ordinary finding, normal polarity"),
        ("Between T1 and T2, the pleural effusion increases.",
         1, WORSENED, "ordinary finding, normal polarity"),
    ]
    for text, ivl, want, why in cases:
        got = option_relations(text).get(ivl)
        assert got == want, f"{why}: {text!r} -> {got}, wanted {want}"

    # subject switches mid-option: clause polarity must not leak across clauses
    mixed = ("Between T1 and T2, lung volumes are normal; from T2 to T3, "
             "lower lobe volume loss diminishes.")
    assert option_relations(mixed).get(2) == IMPROVED, option_relations(mixed)
    return f"polarity: {len(cases) + 1} hand-labelled cases, incl. mid-option subject switch"


def main():
    tests = [
        test_partition_exact,
        test_generator_rows_sum_zero,
        test_ctmc_is_stochastic,
        test_ctmc_semigroup,
        test_trajectory_space,
        test_query_labels_exhaustive,
        test_enumeration_matches_forward_backward,
        test_finding_polarity,
    ]
    print("running correctness checks\n" + "-" * 66)
    for t in tests:
        print(f"  PASS  {t()}")
    print("-" * 66)
    print(f"{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
