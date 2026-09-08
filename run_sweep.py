"""Pilot sweeps. Answers, before any data exists:
   Q1 how accurate must the local comparator be?
   Q2 does global inference beat greedy composition?
"""
import numpy as np

from sim.experiments import METHODS, evaluate, sweep_accuracy, sweep_calibration
from sim.simulate import Config

LABEL = {
    "unary_only": "unary only (no comparator)",
    "greedy": "pairwise + greedy chain",
    "pairwise_only": "pairwise + inference, no unary",
    "no_ctmc": "full, uniform transition prior",
    "no_gate": "full, no reliability gate",
    "naive_missing": "full, 'not mentioned' = absent",
    "full": "FULL MODEL",
}
N = 4000


def table(res, title):
    print(f"\n{title}")
    print("-" * 78)
    print(f"{'method':<34} {'TEL':>7} {'ICR':>7} {'GTS':>7} {'mean':>8} {'consist':>9}")
    print("-" * 78)
    m = res["_majority"]
    print(f"{'majority-class baseline':<34} {m['tel']:>7.3f} {m['icr']:>7.3f} "
          f"{m['gts']:>7.3f} {m['mean']:>8.3f} {'-':>9}")
    for k in METHODS:
        r = res[k]
        star = "  <--" if k == "full" else ""
        print(f"{LABEL[k]:<34} {r['tel']:>7.3f} {r['icr']:>7.3f} {r['gts']:>7.3f} "
              f"{r['mean']:>8.3f} {r['consistency']:>9.3f}{star}")
    print("-" * 78)


def main():
    print("=" * 78)
    print("PILOT SWEEP  -  synthetic, data-independent")
    print(f"5-visit timelines, 4 latent states, {N} trials per cell")
    print("=" * 78)

    base = Config(cmp_accuracy=0.60)
    table(evaluate(base, n_trials=N), "ABLATION at comparator accuracy 0.60")

    print("\n\nQ1  ACCURACY SWEEP  -  mean query accuracy vs comparator accuracy")
    print("-" * 78)
    print(f"{'cmp acc':>8} {'unary':>8} {'greedy':>8} {'full':>8} {'gain':>8} "
          f"{'greedy consist':>16}")
    print("-" * 78)
    rows = sweep_accuracy([0.35, 0.45, 0.55, 0.65, 0.75, 0.85], n_trials=N)
    for a, r in rows:
        g, f = r["greedy"]["mean"], r["full"]["mean"]
        print(f"{a:>8.2f} {r['unary_only']['mean']:>8.3f} {g:>8.3f} {f:>8.3f} "
              f"{f - g:>+8.3f} {r['greedy']['consistency']:>16.3f}")
    print("-" * 78)

    print("\n\nQ2  CALIBRATION SWEEP  -  accuracy fixed at 0.60, probabilities distorted")
    print("-" * 78)
    print(f"{'temp':>8}  {'meaning':<22} {'greedy':>8} {'full':>8} {'gain':>8}")
    print("-" * 78)
    meaning = {0.4: "very overconfident", 0.7: "overconfident",
               1.0: "calibrated", 1.6: "underconfident", 2.5: "very underconfident"}
    for temp, r in sweep_calibration([0.4, 0.7, 1.0, 1.6, 2.5], n_trials=N):
        g, f = r["greedy"]["mean"], r["full"]["mean"]
        print(f"{temp:>8.1f}  {meaning[temp]:<22} {g:>8.3f} {f:>8.3f} {f - g:>+8.3f}")
    print("-" * 78)
    print("\n(greedy uses only argmax, so it is invariant to calibration by construction;\n"
          " any movement in its column is Monte-Carlo noise and indicates the error bar.)")


if __name__ == "__main__":
    main()
