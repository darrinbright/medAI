"""Train the comparator.

    python run_train.py --pairs labelled_pairs.csv --images /data/mimic-cxr-jpg \
                        --null cohort/null_pairs.csv --out runs/comparator

`--pairs` needs path_a, path_b, finding_id, relation, split (and optionally severity_a/_b,
view_a/_b, delta_tau_days). Relations must already be in abnormality-severity polarity - see
docs/micxr-analysis.md, where "lung volume decreased" is a WORSENING.

With no --pairs it trains on a synthetic learnable stand-in, which is how the loop is smoke
tested before data access.
"""
import argparse
import tempfile

import pandas as pd
import torch

from model.comparator import Comparator, ComparatorConfig
from model.data import DataConfig, make_loader, synthetic_pairs
from model.train import TrainConfig, train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs")
    ap.add_argument("--null")
    ap.add_argument("--images", default=".")
    ap.add_argument("--out", default="runs/comparator")
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum-steps", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pretrained", action="store_true")
    a = ap.parse_args()

    if a.pairs:
        df, root = pd.read_csv(a.pairs), a.images
    else:
        root = tempfile.mkdtemp()
        df = synthetic_pairs(root, n_pairs=400, size=min(a.size, 64))
        print(f"no --pairs given; training on a synthetic stand-in in {root}")

    dc = DataConfig(image_root=root, size=a.size, batch_size=a.batch_size,
                    workers=a.workers)
    tr = make_loader(df[df.split == "train"], dc, True, True, seed=0)
    va = make_loader(df[df.split.isin(("validate", "val"))], dc, True, False, seed=1)

    null_loader = None
    if a.null:
        nd = pd.read_csv(a.null)
        nd = nd[nd.split == "train"] if "split" in nd.columns else nd
        if "finding_id" not in nd.columns:
            nd = nd.assign(finding_id=0)     # gate supervision is finding-agnostic
        null_loader = make_loader(nd, dc, False, True, seed=2)

    n_findings = int(df["finding_id"].max()) + 1
    model = Comparator(ComparatorConfig(backbone=a.backbone, n_findings=n_findings,
                                        pretrained=a.pretrained))
    cfg = TrainConfig(epochs=a.epochs, lr=a.lr, accum_steps=a.accum_steps, out_dir=a.out)

    print(f"train {len(tr.dataset)} pairs | val {len(va.dataset)} pairs | "
          f"null {len(null_loader.dataset) if null_loader else 0} | "
          f"findings {n_findings} | device "
          f"{'cuda' if torch.cuda.is_available() else 'cpu'}")
    s = train(model, tr, va, cfg, null_loader=null_loader)
    print(f"\nbest val six-way accuracy {s['best_val_accuracy']:.4f} "
          f"at epoch {s['best_epoch']}  ->  {a.out}/best.pt")


if __name__ == "__main__":
    main()
