"""Stage 1: the local comparator (proposal 3.5).

One finding-conditioned Siamese model producing everything the inference layer consumes:

  relation head    p(relation | x_a, x_b, finding)  -> the pairwise potential
  severity head    p(s | x, finding)                -> the unary term
  reliability head rho(x_a, x_b, acquisition)       -> the gate of proposal 3.6

Design notes that matter:

* **Order is meaningful.** relation(a, b) is not relation(b, a): "new" reverses to "resolved".
  Fusion therefore carries the signed difference f_a - f_b, and a swap-consistency loss ties
  p(r | a, b) to p(inv(r) | b, a). That is free supervision from the relation algebra itself -
  every training pair can be used in both directions.

* **FiLM before pooling**, on the last-stage feature map, so the finding modulates spatial
  features rather than only re-weighting a pooled vector. One model covers all findings, which
  matters because seven findings account for ~98% of MI-CXR (docs/micxr-analysis.md).

* **Grayscale.** Chest radiographs are single-channel; pretrained RGB stems are folded by
  summing over the input-channel axis, which preserves the response to a grey image.

* Relations are defined over **severity of the abnormality**, so training labels for
  volume/aeration findings need polarity conversion first (docs/micxr-analysis.md). That is a
  labelling step, not a model concern - the model never sees the word "decreased".
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Relation indices match sim.core: absent-both, new, resolved, stable, worsened, improved.
N_RELATIONS = 6
N_STATES = 4
RELATION_INVERSE = torch.tensor([0, 2, 1, 3, 5, 4])   # new<->resolved, worsened<->improved

# Acquisition covariates for the reliability head, from the cohort builder's columns.
# view_a and view_b one-hot over (PA, AP), whether the projection changed, and log1p(delta-tau).
N_ACQ_FEATURES = 6


@dataclass
class ComparatorConfig:
    backbone: str = "resnet50"      # resnet50 | resnet18 | tiny
    n_findings: int = 16
    embed_dim: int = 128
    proj_dim: int = 256
    hidden: int = 512
    dropout: float = 0.1
    rho_min: float = 0.05           # gate floor; rho -> 0 would erase evidence entirely
    pretrained: bool = False


# ---------------------------------------------------------------- backbones


class _TinyBackbone(nn.Module):
    """Small CPU-friendly stand-in so the scaffolding is testable without downloads."""

    def __init__(self, out_channels: int = 128):
        super().__init__()
        c = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=2, padding=3), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, c, 3, stride=2, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.out_channels = c

    def forward(self, x):
        return self.net(x)


def _fold_rgb_stem(conv: nn.Conv2d) -> nn.Conv2d:
    """Convert a 3-channel stem to 1 channel by summing weights over the input axis."""
    new = nn.Conv2d(1, conv.out_channels, conv.kernel_size, conv.stride,
                    conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight.copy_(conv.weight.sum(dim=1, keepdim=True))
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def build_backbone(cfg: ComparatorConfig) -> nn.Module:
    """Feature extractor returning a pre-pool map (B, C, H, W)."""
    if cfg.backbone == "tiny":
        return _TinyBackbone()
    from torchvision import models

    ctor = {"resnet50": models.resnet50, "resnet18": models.resnet18}[cfg.backbone]
    net = ctor(weights="IMAGENET1K_V1" if cfg.pretrained else None)
    net.conv1 = _fold_rgb_stem(net.conv1)
    trunk = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                          net.layer1, net.layer2, net.layer3, net.layer4)
    trunk.out_channels = 2048 if cfg.backbone == "resnet50" else 512
    return trunk


# ---------------------------------------------------------------- model


class FiLM(nn.Module):
    """Per-channel affine modulation of a feature map, conditioned on the finding."""

    def __init__(self, embed_dim: int, channels: int):
        super().__init__()
        self.to_gamma = nn.Linear(embed_dim, channels)
        self.to_beta = nn.Linear(embed_dim, channels)
        # NEAR-identity, not exact identity. Biases give gamma=1, beta=0 so training starts
        # essentially unconditioned and pretrained features are not perturbed. The weights get
        # a small non-zero scale rather than zeros: with exact zeros, d(loss)/d(embedding) is
        # identically zero, so the finding embedding receives no gradient on the first step
        # and only begins learning once the FiLM weights have moved. ~1% modulation at init
        # removes that dead pathway at no cost.
        nn.init.normal_(self.to_gamma.weight, std=1e-3); nn.init.ones_(self.to_gamma.bias)
        nn.init.normal_(self.to_beta.weight, std=1e-3); nn.init.zeros_(self.to_beta.bias)

    def forward(self, feat, emb):
        g = self.to_gamma(emb)[:, :, None, None]
        b = self.to_beta(emb)[:, :, None, None]
        return feat * g + b


class Comparator(nn.Module):
    def __init__(self, cfg: ComparatorConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or ComparatorConfig()
        self.backbone = build_backbone(cfg)
        c = self.backbone.out_channels

        self.finding = nn.Embedding(cfg.n_findings, cfg.embed_dim)
        self.film = FiLM(cfg.embed_dim, c)
        self.project = nn.Sequential(nn.Linear(c, cfg.proj_dim), nn.ReLU(inplace=True))

        d = cfg.proj_dim
        self.relation_head = nn.Sequential(
            nn.Linear(4 * d, cfg.hidden), nn.ReLU(inplace=True), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, N_RELATIONS),
        )
        self.severity_head = nn.Sequential(
            nn.Linear(d, cfg.hidden // 2), nn.ReLU(inplace=True), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden // 2, N_STATES),
        )
        self.reliability_head = nn.Sequential(
            nn.Linear(3 * d + N_ACQ_FEATURES, cfg.hidden // 2), nn.ReLU(inplace=True),
            nn.Linear(cfg.hidden // 2, 1),
        )

    # -- pieces ----------------------------------------------------------

    def encode(self, x: torch.Tensor, finding: torch.Tensor) -> torch.Tensor:
        """(B,1,H,W) image + (B,) finding id -> (B, proj_dim) conditioned embedding."""
        emb = self.finding(finding)
        feat = self.film(self.backbone(x), emb)
        return self.project(F.adaptive_avg_pool2d(feat, 1).flatten(1))

    def fuse(self, fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
        """Signed difference makes the fusion order-sensitive, as the relations require."""
        return torch.cat([fa, fb, fa - fb, fa * fb], dim=1)

    # -- forward ---------------------------------------------------------

    def forward(self, x_a, x_b, finding, acq=None, swap=False) -> dict[str, torch.Tensor]:
        """`swap=True` also returns the reversed-pair logits for swap consistency.

        This costs one extra MLP call, not a second backbone pass: the reversed pair reuses
        the SAME encoded features, only fused in the other order. Swap consistency is
        therefore nearly free rather than doubling training cost - which matters on 12 GB,
        where a naive second forward would halve the usable batch size.
        """
        fa = self.encode(x_a, finding)
        fb = self.encode(x_b, finding)
        out = {
            "relation_logits": self.relation_head(self.fuse(fa, fb)),
            "severity_logits_a": self.severity_head(fa),
            "severity_logits_b": self.severity_head(fb),
        }
        if swap:
            out["relation_logits_swapped"] = self.relation_head(self.fuse(fb, fa))
        if acq is None:
            acq = x_a.new_zeros(x_a.shape[0], N_ACQ_FEATURES)
        r = self.reliability_head(torch.cat([fa, fb, (fa - fb).abs(), acq], dim=1))
        # rho in (rho_min, 1]: a wrong gate should damp evidence, never invert it
        out["rho"] = self.cfg.rho_min + (1 - self.cfg.rho_min) * torch.sigmoid(r).squeeze(1)
        return out

    @torch.no_grad()
    def predict(self, x_a, x_b, finding, acq=None) -> dict[str, torch.Tensor]:
        """Exactly what the inference layer consumes: p(relation), p(s), rho."""
        o = self(x_a, x_b, finding, acq)
        return {
            "p_relation": o["relation_logits"].softmax(-1),
            "p_severity_a": o["severity_logits_a"].softmax(-1),
            "p_severity_b": o["severity_logits_b"].softmax(-1),
            "rho": o["rho"],
        }


# ---------------------------------------------------------------- losses


def invert_relations(x: torch.Tensor) -> torch.Tensor:
    """Reindex a (B, 6) relation distribution or logit tensor into the reversed order."""
    return x[:, RELATION_INVERSE.to(x.device)]


def swap_consistency_loss(logits_ab: torch.Tensor, logits_ba: torch.Tensor) -> torch.Tensor:
    """Symmetric KL between p(r | a,b) and inv(p(r | b,a)).

    Free supervision from the relation algebra: reversing a pair must reverse the relation,
    so every labelled pair constrains the model twice. Zero exactly when the two agree.
    """
    p = logits_ab.log_softmax(-1)
    q = invert_relations(logits_ba).log_softmax(-1)
    return 0.5 * (F.kl_div(q, p, log_target=True, reduction="batchmean")
                  + F.kl_div(p, q, log_target=True, reduction="batchmean"))


def null_change_loss(logits_ab: torch.Tensor, allowed=(0, 3)) -> torch.Tensor:
    """Weak null-change supervision (proposal 3.6).

    Same-day pairs are not gold - findings genuinely change within hours - so this asks only
    that mass concentrate on {absent-both, stable} rather than forcing a specific label.
    """
    p = logits_ab.softmax(-1)
    return -(p[:, list(allowed)].sum(-1).clamp_min(1e-8)).log().mean()


def reliability_loss(rho: torch.Tensor, correct: torch.Tensor,
                    rho_min: float) -> torch.Tensor:
    """Train rho to predict whether the comparison is right.

    This gives the gate concrete semantics - rho IS the estimated probability that the
    relation prediction is correct - which is exactly what an inverse-temperature weight
    should be. It also matches what the pilot found (docs/pilot-findings.md): gating is worth
    nothing when mismatch merely adds noise, and pays only when the comparator is CONFIDENTLY
    WRONG. A confidence head trained on correctness is what detects that case.

    The target is detached by construction (an argmax comparison), so gradient flows into rho
    alone and never rewards the relation head for being easy to predict.
    """
    p = ((rho - rho_min) / (1.0 - rho_min)).clamp(1e-6, 1 - 1e-6)
    return F.binary_cross_entropy(p, correct.detach().float())


def comparator_loss(out, batch, w_sev=0.5, w_swap=0.2, w_rho=0.3,
                    rho_min=0.05) -> tuple[torch.Tensor, dict]:
    """Relation CE + severity CE + swap consistency + reliability, with per-term logging."""
    parts = {}
    loss = F.cross_entropy(out["relation_logits"], batch["relation"])
    parts["relation"] = float(loss.detach())

    if "severity_a" in batch:
        sev = 0.5 * (F.cross_entropy(out["severity_logits_a"], batch["severity_a"])
                     + F.cross_entropy(out["severity_logits_b"], batch["severity_b"]))
        loss = loss + w_sev * sev
        parts["severity"] = float(sev.detach())

    if "relation_logits_swapped" in out:
        sw = swap_consistency_loss(out["relation_logits"], out["relation_logits_swapped"])
        loss = loss + w_swap * sw
        parts["swap"] = float(sw.detach())

    if "rho" in out:
        correct = out["relation_logits"].argmax(-1) == batch["relation"]
        rl = reliability_loss(out["rho"], correct, rho_min)
        loss = loss + w_rho * rl
        parts["reliability"] = float(rl.detach())

    parts["total"] = float(loss.detach())
    return loss, parts


def null_change_batch_loss(out, w_rho=0.3, rho_min=0.05,
                           allowed=(0, 3)) -> tuple[torch.Tensor, dict]:
    """Loss for a batch of same-day pairs, which carry no relation label.

    Two signals: mass should sit on {absent-both, stable}, and rho should be low exactly when
    it does not. Same-day pairs are weak supervision - findings do change within hours - so
    nothing here forces a specific relation.
    """
    nc = null_change_loss(out["relation_logits"], allowed)
    parts = {"null_change": float(nc.detach())}
    loss = nc
    if "rho" in out:
        plausible = torch.isin(out["relation_logits"].argmax(-1),
                               torch.tensor(allowed, device=out["rho"].device))
        rl = reliability_loss(out["rho"], plausible, rho_min)
        loss = loss + w_rho * rl
        parts["reliability"] = float(rl.detach())
    parts["total"] = float(loss.detach())
    return loss, parts


def acquisition_features(view_a, view_b, delta_tau_days) -> torch.Tensor:
    """Build the reliability head's covariates from cohort columns.

    view_* are 0 for PA and 1 for AP. Projection change is given explicitly rather than left
    for the network to infer, because it is the covariate proposal 3.6 turns on and it is
    free in the metadata.
    """
    va = F.one_hot(view_a.long(), 2).float()
    vb = F.one_hot(view_b.long(), 2).float()
    changed = (view_a != view_b).float().unsqueeze(1)
    dt = torch.log1p(delta_tau_days.clamp_min(0)).unsqueeze(1)
    return torch.cat([va, vb, changed, dt], dim=1)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone = sum(p.numel() for p in model.backbone.parameters())
    return {"total": total, "trainable": trainable,
            "backbone": backbone, "heads": total - backbone}
