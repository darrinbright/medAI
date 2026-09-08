"""Does the reliability gate earn its place?

The gate is worth nothing when acquisition mismatch merely makes the comparator less
accurate - inference already discounts weak evidence. It only pays when mismatch makes
the comparator CONFIDENTLY WRONG. That is a falsifiable claim about real data.
"""
from sim.experiments import evaluate
from sim.simulate import Config

N = 6000
SCENARIOS = [
    ("less accurate, still calibrated", dict()),
    ("+ confidently wrong when mismatched", dict(cmp_temperature_unreliable=0.30)),
    ("+ confidently wrong, 40% of intervals",
     dict(cmp_temperature_unreliable=0.30, unreliable_rate=0.40)),
    ("+ confidently wrong, 55% of intervals",
     dict(cmp_temperature_unreliable=0.30, unreliable_rate=0.55)),
    ("severe: conf. wrong, 40%, acc 0.22",
     dict(cmp_temperature_unreliable=0.25, unreliable_rate=0.40,
          cmp_accuracy_unreliable=0.22)),
]


def main():
    print("RELIABILITY GATE STUDY  -  mean query accuracy, "
          f"comparator accuracy 0.60, {N} trials")
    print("-" * 76)
    print(f"{'scenario':<46} {'no gate':>9} {'gate':>9} {'gain':>8}")
    print("-" * 76)
    for label, kw in SCENARIOS:
        r = evaluate(Config(cmp_accuracy=0.60, **kw), n_trials=N)
        ng, g = r["no_gate"]["mean"], r["full"]["mean"]
        print(f"{label:<46} {ng:>9.3f} {g:>9.3f} {g - ng:>+8.3f}")
    print("-" * 76)
    print("\nImplication: before claiming the gate as a contribution, measure on real data\n"
          "whether acquisition mismatch produces confident errors or merely weak evidence.")


if __name__ == "__main__":
    main()
