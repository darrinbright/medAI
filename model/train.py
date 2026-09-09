"""Training loop for the comparator.

Handles the two supervision streams together: each step takes a labelled batch and, when
available, a null-change batch, so the reliability head sees same-day pairs throughout rather
than in a separate phase.

Choices worth stating:

* **Early stopping tracks six-way relation accuracy, not loss.** Six-way accuracy is the one
  quantity the whole project turns on (proposal §5), and it is what the end-to-end projection
  is parameterised by. Loss can improve while the argmax does not.

* **Temperature calibration is fitted on validation and shipped with the weights.** The
  inference layer consumes p(relation), not argmax, so an uncalibrated checkpoint is only half
  a deliverable.

* **No class weighting by default.** See data.class_weights - the CTMC prior already applies
  base rates downstream, so a balanced comparator would double-count the correction.

* Gradient accumulation rather than a large batch, because one pair costs two backbone passes
  and 12 GB does not leave much room at 512px.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .comparator import (
    N_RELATIONS, Comparator, ComparatorConfig, acquisition_features, comparator_loss,
    null_change_batch_loss,
)


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-4
    backbone_lr_mult: float = 0.1     # pretrained trunk moves slower than fresh heads
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    accum_steps: int = 4
    grad_clip: float = 1.0
    amp: bool = True
    w_sev: float = 0.5
    w_swap: float = 0.2
    w_rho: float = 0.3
    w_null: float = 0.5
    class_weighting: bool = False
    patience: int = 5
    seed: int = 0
    out_dir: str = "runs/comparator"


# ---------------------------------------------------------------- metrics


def expected_calibration_error(conf, correct, n_bins=15) -> float:
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def macro_f1(pred, true, n=N_RELATIONS) -> float:
    pred, true = np.asarray(pred), np.asarray(true)
    fs = []
    for c in range(n):
        tp = ((pred == c) & (true == c)).sum()
        fp = ((pred == c) & (true != c)).sum()
        fn = ((pred != c) & (true == c)).sum()
        if tp + fn == 0:
            continue                       # class absent from this split
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        fs.append(0.0 if p + r == 0 else 2 * p * r / (p + r))
    return float(np.mean(fs)) if fs else 0.0


# ---------------------------------------------------------------- helpers


def _to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _acq(batch):
    return acquisition_features(batch["view_a"], batch["view_b"], batch["delta_tau"])


def build_optimizer(model, cfg: TrainConfig):
    """Lower LR on the backbone; heads are randomly initialised and need to move faster."""
    bb = [p for n, p in model.named_parameters() if n.startswith("backbone")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("backbone")]
    return torch.optim.AdamW(
        [{"params": bb, "lr": cfg.lr * cfg.backbone_lr_mult},
         {"params": rest, "lr": cfg.lr}], weight_decay=cfg.weight_decay)


def lr_lambda(step: int, total: int, warmup: int):
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


# ---------------------------------------------------------------- evaluation


@torch.no_grad()
def evaluate(model, loader, device, amp=False) -> dict:
    model.eval()
    logits_all, true_all = [], []
    for batch in loader:
        batch = _to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out = model(batch["x_a"], batch["x_b"], batch["finding"], _acq(batch))
        logits_all.append(out["relation_logits"].float().cpu())
        true_all.append(batch["relation"].cpu())
    if not logits_all:
        return {"n": 0, "accuracy": 0.0}

    logits = torch.cat(logits_all)
    true = torch.cat(true_all)
    p = logits.softmax(-1)
    conf, pred = p.max(-1)
    correct = (pred == true)
    return {
        "n": int(len(true)),
        "accuracy": float(correct.float().mean()),
        "macro_f1": macro_f1(pred.numpy(), true.numpy()),
        "nll": float(F.cross_entropy(logits, true)),
        "ece": expected_calibration_error(conf.numpy(), correct.numpy()),
        "_logits": logits, "_true": true,
    }


def fit_temperature(logits: torch.Tensor, true: torch.Tensor) -> float:
    """One scalar T minimising validation NLL (proposal §6). Argmax is unchanged; only the
    sharpness of p(relation) moves, which is what the inference layer actually consumes."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=60)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), true)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().detach())


# ---------------------------------------------------------------- training


def train(model, train_loader, val_loader, cfg: TrainConfig, null_loader=None,
          device=None, log=print) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model.to(device)
    os.makedirs(cfg.out_dir, exist_ok=True)

    amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    opt = build_optimizer(model, cfg)

    steps_per_epoch = max(len(train_loader) // cfg.accum_steps, 1)
    total = steps_per_epoch * cfg.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, total, int(cfg.warmup_frac * total)))

    history, best, best_epoch, bad = [], -1.0, -1, 0
    null_iter = iter(null_loader) if null_loader else None

    for epoch in range(cfg.epochs):
        model.train()
        t0 = time.time()
        running = {}
        opt.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_loader):
            batch = _to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(batch["x_a"], batch["x_b"], batch["finding"], _acq(batch),
                            swap=True)
                loss, parts = comparator_loss(
                    out, batch, cfg.w_sev, cfg.w_swap, cfg.w_rho, model.cfg.rho_min)

                if null_iter is not None:
                    try:
                        nb = next(null_iter)
                    except StopIteration:
                        null_iter = iter(null_loader)
                        nb = next(null_iter)
                    nb = _to_device(nb, device)
                    n_out = model(nb["x_a"], nb["x_b"], nb["finding"], _acq(nb))
                    n_loss, n_parts = null_change_batch_loss(
                        n_out, cfg.w_rho, model.cfg.rho_min)
                    loss = loss + cfg.w_null * n_loss
                    parts["null"] = n_parts["null_change"]

            scaler.scale(loss / cfg.accum_steps).backward()
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + v

            if (i + 1) % cfg.accum_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

        n_batches = max(len(train_loader), 1)
        tr = {k: v / n_batches for k, v in running.items()}
        va = evaluate(model, val_loader, device, amp)
        rec = {"epoch": epoch, "seconds": time.time() - t0, "lr": sched.get_last_lr()[0],
               "train": tr, "val": {k: v for k, v in va.items() if not k.startswith("_")}}
        history.append(rec)
        log(f"epoch {epoch:>3}  loss {tr.get('total', 0):.4f}  "
            f"val acc {va['accuracy']:.4f}  macroF1 {va.get('macro_f1', 0):.4f}  "
            f"ECE {va.get('ece', 0):.4f}  ({rec['seconds']:.1f}s)")

        # early stopping on six-way accuracy - the quantity the project turns on
        if va["accuracy"] > best:
            best, best_epoch, bad = va["accuracy"], epoch, 0
            temperature = fit_temperature(va["_logits"], va["_true"]) if va["n"] else 1.0
            torch.save({"model": model.state_dict(),
                        "comparator_config": asdict(model.cfg),
                        "train_config": asdict(cfg),
                        "temperature": temperature,
                        "val_accuracy": best, "epoch": epoch},
                       os.path.join(cfg.out_dir, "best.pt"))
        else:
            bad += 1
            if bad >= cfg.patience:
                log(f"early stop: no improvement in {cfg.patience} epochs")
                break

    summary = {"best_val_accuracy": best, "best_epoch": best_epoch, "history": history}
    with open(os.path.join(cfg.out_dir, "history.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
