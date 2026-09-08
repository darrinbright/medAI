"""Conditional independence study (proposal 3.9).

Q1  How much accuracy does the violation cost?
Q2  Does it make the posterior overconfident, and by how much?
Q3  Does likelihood tempering - one shared exponent on all image-derived evidence -
    recover calibration, and at what cost to accuracy?
"""
import numpy as np

from sim.core import (
    REL, T, log_posterior, onset, query_gts, query_icr, query_tel, shape,
)
from sim.experiments import RHO_LOW, greedy_chain
from sim.depend import expected_calibration_error, simulate_correlated
from sim.simulate import Config, log_unary_from_obs, sample_timeline

N = 5000
QUERIES = ("tel", "gts", "icr")


def run(cfg, strength, weight, rng):
    """weight w tempers all image-derived evidence; the CTMC prior keeps full weight."""
    hits = {"full": {q: 0 for q in QUERIES}, "greedy": {q: 0 for q in QUERIES}}
    conf, corr = [], []

    for _ in range(N):
        s, dts, Ps = sample_timeline(cfg, rng)
        y, p_cmp, reliable, _ = simulate_correlated(s, cfg, strength, rng)

        log_cmp = np.log(np.clip(p_cmp, 1e-12, None))
        log_ctmc = np.log(np.clip(np.stack(Ps), 1e-12, None))
        lu = log_unary_from_obs(y, cfg, missingness_aware=True)
        gate = np.where(reliable, 1.0, RHO_LOW)
        log_pi = np.log(np.array(cfg.pi))

        post = log_posterior(weight * lu, log_cmp, log_ctmc, weight * gate, log_pi)

        i = int(rng.integers(0, T - 1))
        j = int(rng.integers(i + 1, T))
        truth = {"tel": int(onset(s[None, :])[0]), "gts": int(shape(s[None, :])[0]),
                 "icr": int(REL[s[i], s[j]])}

        q_tel = query_tel(post)
        pred = {"tel": int(np.argmax(q_tel)), "gts": int(np.argmax(query_gts(post))),
                "icr": int(np.argmax(query_icr(post, i, j)))}
        for q in QUERIES:
            hits["full"][q] += int(pred[q] == truth[q])
        conf.append(float(q_tel.max()))
        corr.append(float(pred["tel"] == truth["tel"]))

        s_g, _ = greedy_chain(y, p_cmp, Ps, cfg)
        pg = {"tel": int(onset(s_g[None, :])[0]), "gts": int(shape(s_g[None, :])[0]),
              "icr": int(REL[s_g[i], s_g[j]])}
        for q in QUERIES:
            hits["greedy"][q] += int(pg[q] == truth[q])

    ece, gap = expected_calibration_error(np.array(conf), np.array(corr))
    return {
        "full": float(np.mean([hits["full"][q] / N for q in QUERIES])),
        "greedy": float(np.mean([hits["greedy"][q] / N for q in QUERIES])),
        "ece": ece,
        "gap": gap,
    }


def main():
    cfg = Config(cmp_accuracy=0.60)
    print("=" * 82)
    print("CONDITIONAL INDEPENDENCE STUDY  (proposal 3.9)")
    print("A per-image nuisance factor shifts apparent severity, corrupting the unary")
    print(f"read and BOTH pairwise comparisons touching that image.  {N} trials per cell.")
    print("=" * 82)

    print("\nQ1/Q2  cost of the violation, no mitigation (w = 1.0)")
    print("-" * 82)
    print(f"{'shared nuisance':>16} {'greedy':>9} {'full':>9} {'gain':>8} "
          f"{'TEL ECE':>9} {'overconf':>10}")
    print("-" * 82)
    strengths = [0.0, 0.10, 0.20, 0.35, 0.50]
    base = {}
    for st in strengths:
        r = run(cfg, st, 1.0, np.random.default_rng(1000 + int(st * 100)))
        base[st] = r
        print(f"{st:>16.2f} {r['greedy']:>9.3f} {r['full']:>9.3f} "
              f"{r['full'] - r['greedy']:>+8.3f} {r['ece']:>9.3f} {r['gap']:>+10.3f}")
    print("-" * 82)

    print("\nQ3  likelihood tempering at shared nuisance = 0.35")
    print("-" * 82)
    print(f"{'weight w':>10} {'full acc':>10} {'TEL ECE':>9} {'overconf':>10}")
    print("-" * 82)
    for w in (1.0, 0.85, 0.7, 0.55, 0.4):
        r = run(cfg, 0.35, w, np.random.default_rng(7000 + int(w * 100)))
        print(f"{w:>10.2f} {r['full']:>10.3f} {r['ece']:>9.3f} {r['gap']:>+10.3f}")
    print("-" * 82)
    print("\nPositive 'overconf' means the posterior claims more confidence than it earns.")


if __name__ == "__main__":
    main()
