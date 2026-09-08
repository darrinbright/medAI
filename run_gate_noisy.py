"""How good must a LEARNED reliability gate be before it earns its place?

Every gate number so far used an oracle that knows which intervals are unreliable, so those
gains are upper bounds. Here the gate is a classifier of a given AUC: it sees a score that
separates reliable from unreliable intervals by d = sqrt(2) * Phi^-1(AUC), thresholds it, and
tempers the comparator evidence accordingly.

Run under the only condition where the gate pays at all (proposal 3.6): acquisition mismatch
makes the comparator confidently wrong, not merely uncertain.
"""
import numpy as np
from scipy.stats import norm

from sim.core import REL, T, log_posterior, onset, query_gts, query_icr, query_tel, shape
from sim.experiments import RHO_LOW
from sim.simulate import (
    Config, log_unary_from_obs, sample_timeline, simulate_comparator, simulate_unary,
)

N = 5000
QUERIES = ("tel", "gts", "icr")


def gate_scores(reliable, auc, rng):
    """Classifier scores with the requested separability; threshold at the midpoint."""
    d = np.sqrt(2.0) * norm.ppf(np.clip(auc, 0.5001, 0.9999))
    return rng.normal(0.0, 1.0, size=reliable.shape) + d * reliable, d / 2.0


def run(cfg, auc, rng):
    hits = {q: 0 for q in QUERIES}
    tp = fp = tn = fn = 0
    for _ in range(N):
        s, dts, Ps = sample_timeline(cfg, rng)
        p_cmp, reliable = simulate_comparator(s, cfg, rng)
        y = simulate_unary(s, cfg, rng)

        if auc is None:                      # oracle
            gate = np.where(reliable, 1.0, RHO_LOW)
        elif auc <= 0.5:                     # no gate at all
            gate = np.ones(T - 1)
        else:
            sc, thr = gate_scores(reliable.astype(float), auc, rng)
            pred = sc > thr
            gate = np.where(pred, 1.0, RHO_LOW)
            tp += int((pred & reliable).sum()); fp += int((pred & ~reliable).sum())
            tn += int((~pred & ~reliable).sum()); fn += int((~pred & reliable).sum())

        log_cmp = np.log(np.clip(p_cmp, 1e-12, None))
        log_ctmc = np.log(np.clip(np.stack(Ps), 1e-12, None))
        lu = log_unary_from_obs(y, cfg, missingness_aware=True)
        post = log_posterior(lu, log_cmp, log_ctmc, gate, np.log(np.array(cfg.pi)))

        i = int(rng.integers(0, T - 1)); j = int(rng.integers(i + 1, T))
        truth = {"tel": int(onset(s[None, :])[0]), "gts": int(shape(s[None, :])[0]),
                 "icr": int(REL[s[i], s[j]])}
        pred_q = {"tel": int(np.argmax(query_tel(post))),
                  "gts": int(np.argmax(query_gts(post))),
                  "icr": int(np.argmax(query_icr(post, i, j)))}
        for q in QUERIES:
            hits[q] += int(pred_q[q] == truth[q])

    acc = float(np.mean([hits[q] / N for q in QUERIES]))
    detect = (tp + tn) / max(tp + tn + fp + fn, 1)
    return acc, detect


def main():
    cfg = Config(cmp_accuracy=0.60, cmp_temperature_unreliable=0.30,
                 unreliable_rate=0.40)
    print("=" * 76)
    print("LEARNED GATE STUDY")
    print("condition: mismatch makes the comparator CONFIDENTLY WRONG, 40% of intervals")
    print(f"(the only regime where the gate pays at all).  {N} trials per row.")
    print("=" * 76)
    print(f"{'gate AUC':>10} {'interval acc':>14} {'query acc':>11} {'vs no gate':>12}")
    print("-" * 76)

    base, _ = run(cfg, 0.5, np.random.default_rng(11))
    print(f"{'none':>10} {'-':>14} {base:>11.3f} {0.0:>+12.3f}")
    for auc in (0.60, 0.70, 0.80, 0.90, 0.95):
        acc, det = run(cfg, auc, np.random.default_rng(int(auc * 1000)))
        print(f"{auc:>10.2f} {det:>14.3f} {acc:>11.3f} {acc - base:>+12.3f}")
    orc, _ = run(cfg, None, np.random.default_rng(99))
    print(f"{'oracle':>10} {1.0:>14.3f} {orc:>11.3f} {orc - base:>+12.3f}")
    print("-" * 76)
    print("\nRead the break-even AUC off the last column: below it, a learned gate is")
    print("not worth building; above it, report the gate with its detector's AUC.")


if __name__ == "__main__":
    main()
