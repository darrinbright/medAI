"""Misspecification study: is the CTMC gain real, or the simulator agreeing with itself?

For each generative process:
  1. simulate a FIT set, estimate Q by MLE (proposal 3.7's open question)
  2. simulate a held-out EVAL set from the same process
  3. run inference with the fitted - and for non-CTMC processes, wrong - prior
"""
import numpy as np

from sim.core import (
    REL, T, onset, query_gts, query_icr, query_tel, shape, log_posterior,
    transition_matrix,
)
from sim.experiments import RHO_LOW, greedy_chain
from sim.misspec import PROCESS_VIOLATION, PROCESSES, fit_ctmc_mle, simulate_paths
from sim.simulate import (
    Config, log_unary_from_obs, simulate_comparator, simulate_unary,
)

N_FIT, N_EVAL = 6000, 4000
QUERIES = ("tel", "gts", "icr")


def eval_process(process, cfg, rng):
    fit_states, fit_dts = simulate_paths(process, N_FIT, cfg, rng)
    Q_hat, rates = fit_ctmc_mle(fit_states, fit_dts)

    states, dts = simulate_paths(process, N_EVAL, cfg, rng)
    log_pi = np.log(np.clip(np.bincount(states[:, 0], minlength=4) / N_EVAL, 1e-6, None))
    uniform_log_ctmc = np.full((T - 1, 4, 4), np.log(0.25))

    hits = {m: {q: 0 for q in QUERIES} for m in ("greedy", "no_ctmc", "full")}
    cons = {"greedy": 0, "no_ctmc": 0, "full": 0}

    for n in range(N_EVAL):
        s, dt = states[n], dts[n]
        Ps = [transition_matrix(Q_hat, float(d)) for d in dt]
        log_ctmc = np.log(np.clip(np.stack(Ps), 1e-12, None))

        p_cmp, reliable = simulate_comparator(s, cfg, rng)
        y = simulate_unary(s, cfg, rng)
        log_cmp = np.log(np.clip(p_cmp, 1e-12, None))
        lu = log_unary_from_obs(y, cfg, missingness_aware=True)
        gate = np.where(reliable, 1.0, RHO_LOW)

        i = int(rng.integers(0, T - 1))
        j = int(rng.integers(i + 1, T))
        truth = {"tel": int(onset(s[None, :])[0]), "gts": int(shape(s[None, :])[0]),
                 "icr": int(REL[s[i], s[j]])}

        for name, lg in (("full", log_ctmc), ("no_ctmc", uniform_log_ctmc)):
            post = log_posterior(lu, log_cmp, lg, gate, log_pi)
            pred = {"tel": int(np.argmax(query_tel(post))),
                    "gts": int(np.argmax(query_gts(post))),
                    "icr": int(np.argmax(query_icr(post, i, j)))}
            for q in QUERIES:
                hits[name][q] += int(pred[q] == truth[q])
            cons[name] += 1

        s_g, bad = greedy_chain(y, p_cmp, Ps, cfg)
        pred = {"tel": int(onset(s_g[None, :])[0]), "gts": int(shape(s_g[None, :])[0]),
                "icr": int(REL[s_g[i], s_g[j]])}
        for q in QUERIES:
            hits["greedy"][q] += int(pred[q] == truth[q])
        cons["greedy"] += 0 if bad else 1

    out = {m: float(np.mean([hits[m][q] / N_EVAL for q in QUERIES]))
           for m in hits}
    out["greedy_consistency"] = cons["greedy"] / N_EVAL
    out["rates"] = rates
    return out


def main():
    cfg = Config(cmp_accuracy=0.60)
    rng = np.random.default_rng(20260908)

    print("=" * 84)
    print("MISSPECIFICATION STUDY")
    print(f"Q fitted by MLE to each process ({N_FIT} timelines), "
          f"evaluated held-out ({N_EVAL})")
    print("=" * 84)
    print(f"{'process':<15} {'violates':<24} {'greedy':>8} {'uniform':>9} "
          f"{'fitted Q':>9} {'CTMC gain':>10}")
    print("-" * 84)

    results = {}
    for proc in PROCESSES:
        r = eval_process(proc, cfg, rng)
        results[proc] = r
        gain = r["full"] - r["no_ctmc"]
        print(f"{proc:<15} {PROCESS_VIOLATION[proc]:<24} {r['greedy']:>8.3f} "
              f"{r['no_ctmc']:>9.3f} {r['full']:>9.3f} {gain:>+10.3f}")
    print("-" * 84)

    print(f"\n{'process':<15} {'full vs greedy':>15} {'greedy consistency':>20}")
    print("-" * 84)
    for proc in PROCESSES:
        r = results[proc]
        print(f"{proc:<15} {r['full'] - r['greedy']:>+15.3f} "
              f"{r['greedy_consistency']:>20.3f}")
    print("-" * 84)

    print("\nfitted birth-death rates (1/day)  q01    q12    q23    q10    q21    q32")
    print("-" * 84)
    true = np.array(cfg.up + cfg.down)
    print(f"{'true (ctmc)':<22}" + "".join(f"{v:>7.4f}" for v in true))
    for proc in PROCESSES:
        print(f"{proc:<22}" + "".join(f"{v:>7.4f}" for v in results[proc]["rates"]))
    print("-" * 84)


if __name__ == "__main__":
    main()
