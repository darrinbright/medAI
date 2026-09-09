"""Probe the comparator: parameter budget, and the largest batch that fits.

On a CUDA machine this runs real forward+backward steps at increasing batch sizes and
reports peak memory, which is the only honest way to size a batch for a 12 GB card. On CPU it
verifies the step works and reports parameter counts and per-step timing.

    python run_comparator_probe.py --backbone resnet50 --size 512
"""
import argparse
import time

import torch

from model.comparator import (
    Comparator, ComparatorConfig, comparator_loss, count_parameters,
)


def one_step(model, opt, batch_pairs, size, device, amp):
    xa = torch.randn(batch_pairs, 1, size, size, device=device)
    xb = torch.randn(batch_pairs, 1, size, size, device=device)
    f = torch.randint(0, model.cfg.n_findings, (batch_pairs,), device=device)
    tgt = {
        "relation": torch.randint(0, 6, (batch_pairs,), device=device),
        "severity_a": torch.randint(0, 4, (batch_pairs,), device=device),
        "severity_b": torch.randint(0, 4, (batch_pairs,), device=device),
    }
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        out = model(xa, xb, f, swap=True)
        loss, parts = comparator_loss(out, tgt, rho_min=model.cfg.rho_min)
    loss.backward()
    opt.step()
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--no-amp", action="store_true")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not a.no_amp) and device.type == "cuda"
    cfg = ComparatorConfig(backbone=a.backbone)
    model = Comparator(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    n = count_parameters(model)
    print("=" * 70)
    print(f"COMPARATOR PROBE   backbone={a.backbone} size={a.size} device={device.type} "
          f"amp={amp}")
    print("=" * 70)
    print(f"\nparameters      {n['total'] / 1e6:.1f}M total")
    print(f"                {n['backbone'] / 1e6:.1f}M backbone + {n['heads'] / 1e6:.1f}M heads")
    fp32 = n["total"] * 4 / 1e9
    print(f"weights (fp32)  {fp32:.2f} GB")
    print(f"+ AdamW state   {fp32 * 3:.2f} GB   (weights + 2 moments)")
    print("\nnote: one pair costs TWO backbone passes; swap consistency adds only an MLP call.")

    if device.type == "cpu":
        t0 = time.time()
        parts = one_step(model, opt, 2, min(a.size, 128), device, False)
        print(f"\nCPU step ok at batch 2, size {min(a.size, 128)}: "
              f"{time.time() - t0:.1f}s, loss {parts['total']:.3f}")
        print("terms: " + ", ".join(f"{k}={v:.3f}" for k, v in parts.items() if k != "total"))
        print("\nRun this on the training machine for real batch sizing.")
        return

    print(f"\n{'pairs':>7} {'images':>7} {'peak GB':>9} {'s/step':>8} {'status':>9}")
    print("-" * 70)
    best = 0
    b = 1
    while b <= a.max_batch:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            one_step(model, opt, b, a.size, device, amp)     # warm up
            torch.cuda.synchronize()
            t0 = time.time()
            one_step(model, opt, b, a.size, device, amp)
            torch.cuda.synchronize()
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1e9
            print(f"{b:>7} {2 * b:>7} {peak:>9.2f} {dt:>8.2f} {'ok':>9}")
            best = b
        except torch.cuda.OutOfMemoryError:
            print(f"{b:>7} {2 * b:>7} {'-':>9} {'-':>8} {'OOM':>9}")
            break
        b *= 2
    print("-" * 70)
    print(f"largest power-of-two batch that fits: {best} pairs ({2 * best} images)")
    print("Use gradient accumulation to reach the effective batch you want.")


if __name__ == "__main__":
    main()
