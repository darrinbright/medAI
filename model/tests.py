"""Correctness checks for the comparator scaffolding."""
from __future__ import annotations

import itertools

import torch

from .comparator import (
    N_ACQ_FEATURES, N_RELATIONS, N_STATES, RELATION_INVERSE, Comparator,
    ComparatorConfig, acquisition_features, comparator_loss, count_parameters,
    invert_relations, null_change_batch_loss, null_change_loss, reliability_loss,
    swap_consistency_loss,
)

CFG = ComparatorConfig(backbone="tiny", n_findings=8, proj_dim=64, hidden=64)


def _batch(b=3, size=64):
    torch.manual_seed(0)
    return (torch.randn(b, 1, size, size), torch.randn(b, 1, size, size),
            torch.randint(0, CFG.n_findings, (b,)))


def test_relation_inverse_matches_algebra():
    """The model's inverse table must agree with the relation partition in sim.core.

    Ties the swap-consistency loss to the algebra rather than to a hand-typed constant: for
    every state pair, reversing the pair must reverse the relation.
    """
    from sim.core import K, relation

    assert RELATION_INVERSE[RELATION_INVERSE].tolist() == list(range(N_RELATIONS))
    for s, sp in itertools.product(range(K), repeat=2):
        assert RELATION_INVERSE[relation(s, sp)].item() == relation(sp, s), (s, sp)
    return "comparator: relation inverse is an involution and matches sim.core exactly"


def test_shapes_and_ranges():
    m = Comparator(CFG).eval()
    xa, xb, f = _batch()
    out = m(xa, xb, f)
    assert out["relation_logits"].shape == (3, N_RELATIONS)
    assert out["severity_logits_a"].shape == (3, N_STATES)
    assert out["severity_logits_b"].shape == (3, N_STATES)
    assert out["rho"].shape == (3,)
    assert (out["rho"] > CFG.rho_min - 1e-6).all() and (out["rho"] <= 1.0).all()

    p = m.predict(xa, xb, f)
    assert torch.allclose(p["p_relation"].sum(-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(p["p_severity_a"].sum(-1), torch.ones(3), atol=1e-5)
    return "comparator: head shapes correct, rho bounded in (rho_min, 1], outputs normalised"


def test_order_sensitivity():
    """Swapping the pair must change the prediction - "new" and "resolved" are not the same."""
    m = Comparator(CFG).eval()
    xa, xb, f = _batch()
    with torch.no_grad():
        ab = m(xa, xb, f)["relation_logits"]
        ba = m(xb, xa, f)["relation_logits"]
    assert not torch.allclose(ab, ba, atol=1e-3), "fusion is order-invariant"

    # swap=True must reuse the same features rather than re-encoding: identical to the
    # explicit reversed call, at the cost of one extra MLP pass
    with torch.no_grad():
        cheap = m(xa, xb, f, swap=True)["relation_logits_swapped"]
    assert torch.allclose(cheap, ba, atol=1e-5), "cheap swap disagrees with a full reverse pass"
    return "comparator: order-sensitive fusion; cheap swap matches a full reversed pass"


def test_finding_conditioning():
    """FiLM starts NEAR identity, and the conditioning pathway must be live from step 0.

    Biases give gamma=1, beta=0 so training begins essentially unconditioned. Weights carry a
    small non-zero scale: with exact zeros the finding embedding gets identically zero
    gradient on the first step, since d(loss)/d(embedding) passes through those weights.

    So the assertion is not "different findings give different outputs" on an untrained model
    - at init they should be nearly the same, deliberately - but that the pathway is live and
    the embedding is learning immediately.
    """
    m = Comparator(CFG).eval()
    xa, xb, _ = _batch()
    f0 = torch.zeros(3, dtype=torch.long)
    f1 = torch.ones(3, dtype=torch.long)

    with torch.no_grad():
        a0 = m(xa, xb, f0)["relation_logits"]
        b0 = m(xa, xb, f1)["relation_logits"]
    spread = (a0 - b0).abs().max() / a0.abs().max().clamp_min(1e-6)
    assert spread < 0.1, f"FiLM should start near identity, relative spread {spread:.3f}"

    with torch.no_grad():                       # simulate a trained FiLM
        m.film.to_gamma.weight.normal_(0, 0.05)
        m.film.to_beta.weight.normal_(0, 0.05)
        a1 = m(xa, xb, f0)["relation_logits"]
        b1 = m(xa, xb, f1)["relation_logits"]
    assert not torch.allclose(a1, b1, atol=1e-4), "conditioning pathway is dead"

    # and the embedding itself must start learning once FiLM weights are non-zero
    m.zero_grad(set_to_none=True)
    m.train()
    out = m(xa, xb, f0)
    F_ = torch.nn.functional
    F_.cross_entropy(out["relation_logits"], torch.zeros(3, dtype=torch.long)).backward()
    g = m.finding.weight.grad
    assert g is not None and g.abs().sum() > 0, "finding embedding receives no gradient"
    return "comparator: FiLM is identity at init, conditioning learnable, embedding gets gradient"


def test_swap_consistency_loss():
    """Zero exactly when p(r|a,b) equals inv(p(r|b,a)); positive otherwise."""
    torch.manual_seed(1)
    logits = torch.randn(4, N_RELATIONS)
    perfect = invert_relations(logits)                    # the consistent counterpart
    assert float(swap_consistency_loss(logits, perfect)) < 1e-6
    assert float(swap_consistency_loss(logits, torch.randn(4, N_RELATIONS))) > 1e-3

    # a distribution that says "new" should reverse to one that says "resolved"
    p = torch.zeros(1, N_RELATIONS); p[0, 1] = 10.0       # new
    assert int(invert_relations(p).argmax()) == 2         # resolved
    return "comparator: swap-consistency loss is zero iff the reversed pair agrees"


def test_null_change_loss():
    """Weak supervision: reward mass on {absent-both, stable}, do not force a label."""
    stable = torch.full((2, N_RELATIONS), -10.0); stable[:, 3] = 10.0
    worse = torch.full((2, N_RELATIONS), -10.0); worse[:, 4] = 10.0
    assert float(null_change_loss(stable)) < 1e-3
    assert float(null_change_loss(worse)) > 1.0
    absent = torch.full((2, N_RELATIONS), -10.0); absent[:, 0] = 10.0
    assert float(null_change_loss(absent)) < 1e-3        # both allowed labels score well
    return "comparator: null-change loss rewards {absent-both, stable} without forcing one"


def test_gradients_flow():
    m = Comparator(CFG).train()
    xa, xb, f = _batch()
    out = m(xa, xb, f, swap=True)
    batch = {"relation": torch.randint(0, N_RELATIONS, (3,)),
             "severity_a": torch.randint(0, N_STATES, (3,)),
             "severity_b": torch.randint(0, N_STATES, (3,))}
    loss, parts = comparator_loss(out, batch, rho_min=CFG.rho_min)
    loss.backward()
    dead = [n for n, p in m.named_parameters()
            if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert not dead, f"no/!finite gradient: {dead[:5]}"
    assert set(parts) == {"relation", "severity", "swap", "reliability", "total"}
    # every head must be reached, the reliability head included - it was silently
    # untrainable until rho was given its own objective
    for head in ("relation_head", "severity_head", "reliability_head", "film", "finding"):
        g = [p.grad.abs().sum() for n, p in m.named_parameters() if n.startswith(head)]
        assert g and sum(g) > 0, f"{head} received only zero gradient"
    return f"comparator: gradients reach all {sum(1 for _ in m.parameters())} parameter tensors"


def test_reliability_semantics():
    """rho is the estimated probability that the comparison is correct."""
    rho_hi = torch.tensor([0.95, 0.95]); rho_lo = torch.tensor([0.10, 0.10])
    right = torch.tensor([True, True]); wrong = torch.tensor([False, False])
    assert float(reliability_loss(rho_hi, right, 0.05)) < float(
        reliability_loss(rho_lo, right, 0.05))
    assert float(reliability_loss(rho_lo, wrong, 0.05)) < float(
        reliability_loss(rho_hi, wrong, 0.05))

    # unlabelled same-day pairs still train both the relation head and the gate
    m = Comparator(CFG).train()
    xa, xb, f = _batch()
    loss, parts = null_change_batch_loss(m(xa, xb, f), rho_min=CFG.rho_min)
    loss.backward()
    assert set(parts) == {"null_change", "reliability", "total"}
    assert sum(p.grad.abs().sum() for n, p in m.named_parameters()
               if n.startswith("reliability_head")) > 0
    return "comparator: rho trained as P(prediction correct); null pairs train it unlabelled"


def test_acquisition_features():
    va = torch.tensor([0, 1, 1]); vb = torch.tensor([0, 0, 1])
    dt = torch.tensor([0.0, 3.0, 400.0])
    acq = acquisition_features(va, vb, dt)
    assert acq.shape == (3, N_ACQ_FEATURES)
    assert acq[:, 4].tolist() == [0.0, 1.0, 0.0]          # projection-change flag
    assert acq[0, 5] == 0.0 and acq[2, 5] > acq[1, 5]     # log1p(delta-tau), monotone
    return "comparator: acquisition features encode projection change and log delta-tau"


def main():
    tests = [
        test_relation_inverse_matches_algebra,
        test_shapes_and_ranges,
        test_order_sensitivity,
        test_finding_conditioning,
        test_swap_consistency_loss,
        test_null_change_loss,
        test_gradients_flow,
        test_reliability_semantics,
        test_acquisition_features,
    ]
    print("comparator checks\n" + "-" * 70)
    for t in tests:
        print(f"  PASS  {t()}")
    print("-" * 70)
    n = count_parameters(Comparator(ComparatorConfig(backbone="resnet50")))
    print(f"resnet50 config: {n['total'] / 1e6:.1f}M params "
          f"({n['backbone'] / 1e6:.1f}M backbone + {n['heads'] / 1e6:.1f}M heads)")
    print(f"{len(tests)}/{len(tests)} passed")


if __name__ == "__main__":
    main()
