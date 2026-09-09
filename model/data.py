"""Datasets for the comparator: labelled relation pairs and unlabelled null-change pairs.

Two streams, because they carry different supervision:

  labelled   (image_a, image_b, finding, relation[, severity_a, severity_b])
             from Chest ImaGenome comparison relations, converted to abnormality-severity
             polarity first (docs/micxr-analysis.md).

  null       (image_a, image_b, finding) from the cohort builder's same-day pairs. No relation
             label - these are WEAK supervision, so the loss only asks for mass on
             {absent-both, stable} and trains the gate on whether that holds.

Augmentation is applied INDEPENDENTLY to the two images of a pair, and deliberately kept mild.
Real follow-up radiographs genuinely differ in positioning, rotation and exposure, so shared
transforms would train a comparator that never sees the nuisance it must be robust to. Push it
too far and the change signal itself is destroyed, which is why the defaults are small.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# MIMIC-CXR-JPG grayscale statistics; close enough for standardisation.
PIXEL_MEAN, PIXEL_STD = 0.485, 0.229

VIEW_ID = {"PA": 0, "AP": 1}


@dataclass
class DataConfig:
    image_root: str = "."
    size: int = 512
    augment: bool = True
    max_rotate_deg: float = 5.0
    max_translate: float = 0.03
    brightness: float = 0.10
    contrast: float = 0.10
    batch_size: int = 8
    null_batch_size: int = 4
    workers: int = 4


def _load_image(path: str, size: int) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def _augment(x: np.ndarray, cfg: DataConfig, rng: np.random.Generator) -> np.ndarray:
    """Mild, independent per-image jitter standing in for acquisition variation."""
    if not cfg.augment:
        return x
    t = torch.from_numpy(x)[None, None]
    ang = float(rng.uniform(-cfg.max_rotate_deg, cfg.max_rotate_deg))
    dx, dy = rng.uniform(-cfg.max_translate, cfg.max_translate, size=2)
    theta = torch.tensor([[
        [np.cos(np.deg2rad(ang)), -np.sin(np.deg2rad(ang)), float(dx)],
        [np.sin(np.deg2rad(ang)), np.cos(np.deg2rad(ang)), float(dy)],
    ]], dtype=torch.float32)
    grid = torch.nn.functional.affine_grid(theta, t.shape, align_corners=False)
    t = torch.nn.functional.grid_sample(t, grid, align_corners=False, padding_mode="border")
    out = t[0, 0].numpy()
    out = out * float(rng.uniform(1 - cfg.contrast, 1 + cfg.contrast))
    out = out + float(rng.uniform(-cfg.brightness, cfg.brightness))
    return np.clip(out, 0.0, 1.0)


class PairDataset(Dataset):
    """Rows need path_a, path_b, finding_id; labelled rows also need `relation`."""

    def __init__(self, df: pd.DataFrame, cfg: DataConfig, labelled: bool, seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.labelled = labelled
        self.seed = seed
        if labelled and "relation" not in df.columns:
            raise ValueError("labelled dataset needs a `relation` column")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        r = self.df.iloc[i]
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        root = self.cfg.image_root
        xa = _augment(_load_image(os.path.join(root, r["path_a"]), self.cfg.size), self.cfg, rng)
        xb = _augment(_load_image(os.path.join(root, r["path_b"]), self.cfg.size), self.cfg, rng)

        item = {
            "x_a": torch.from_numpy((xa - PIXEL_MEAN) / PIXEL_STD)[None],
            "x_b": torch.from_numpy((xb - PIXEL_MEAN) / PIXEL_STD)[None],
            "finding": torch.tensor(int(r["finding_id"]), dtype=torch.long),
            "view_a": torch.tensor(VIEW_ID.get(str(r.get("view_a", "AP")), 1)),
            "view_b": torch.tensor(VIEW_ID.get(str(r.get("view_b", "AP")), 1)),
            "delta_tau": torch.tensor(float(r.get("delta_tau_days", 0.0) or 0.0)),
        }
        if self.labelled:
            item["relation"] = torch.tensor(int(r["relation"]), dtype=torch.long)
            if "severity_a" in r and not pd.isna(r["severity_a"]):
                item["severity_a"] = torch.tensor(int(r["severity_a"]), dtype=torch.long)
                item["severity_b"] = torch.tensor(int(r["severity_b"]), dtype=torch.long)
        return item


def make_loader(df, cfg: DataConfig, labelled: bool, shuffle: bool, seed: int = 0,
                batch_size: int | None = None) -> DataLoader:
    return DataLoader(
        PairDataset(df, cfg, labelled, seed),
        batch_size=batch_size or (cfg.batch_size if labelled else cfg.null_batch_size),
        shuffle=shuffle, num_workers=cfg.workers, drop_last=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


def class_weights(df: pd.DataFrame, n_relations: int = 6) -> torch.Tensor:
    """Inverse-frequency weights.

    Available, but OFF by default in training. The inference layer consumes p(relation) and
    already applies its own base rates through the CTMC prior, so a comparator reporting
    balanced rather than honest posteriors would double-count the correction and damage the
    calibration the pilot showed is worth ~5 points.
    """
    counts = np.bincount(df["relation"].to_numpy(), minlength=n_relations).astype(float)
    w = counts.sum() / np.maximum(counts, 1.0)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


# ---------------------------------------------------------------- synthetic stand-in


def synthetic_pairs(out_dir: str, n_pairs: int = 96, size: int = 64,
                    n_findings: int = 4, seed: int = 0):
    """A small, genuinely LEARNABLE task: severity is encoded as blob radius.

    Random-noise images would let the training loop "run" while proving nothing. Here the
    relation follows deterministically from the two severities, so a working loop must drive
    accuracy above chance - which is what the smoke test asserts.
    """
    from sim.core import relation

    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    yy, xx = np.mgrid[0:size, 0:size]
    rows = []

    def render(sev: int, path: str):
        img = rng.normal(0.45, 0.03, (size, size))
        if sev > 0:
            cy, cx = size // 2, size // 2
            rad = 3 + 4 * sev
            img += 0.35 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * rad ** 2)))
        Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)

    for i in range(n_pairs):
        sa, sb = int(rng.integers(0, 4)), int(rng.integers(0, 4))
        pa, pb = f"a{i}.png", f"b{i}.png"
        render(sa, os.path.join(out_dir, pa))
        render(sb, os.path.join(out_dir, pb))
        rows.append({
            "path_a": pa, "path_b": pb,
            "finding_id": int(rng.integers(0, n_findings)),
            "relation": relation(sa, sb), "severity_a": sa, "severity_b": sb,
            "view_a": rng.choice(["PA", "AP"]), "view_b": rng.choice(["PA", "AP"]),
            "delta_tau_days": float(np.exp(rng.uniform(0, 5))),
            "split": "train" if i < int(0.7 * n_pairs) else "validate",
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "pairs.csv"), index=False)
    return df
