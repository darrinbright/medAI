"""Build the five-visit cohort, null-change pairs and download manifest.

    python run_cohort.py --metadata mimic-cxr-2.0.0-metadata.csv.gz \
                         --split mimic-cxr-2.0.0-split.csv.gz --out cohort/

With no --metadata it runs on a synthetic stand-in with the same schema, so the pipeline is
exercisable before PhysioNet access arrives.
"""
import argparse
import pathlib

import pandas as pd

from sim.cohort import (
    CohortConfig, build_null_pairs, build_timelines, load_metadata, manifest,
    synthetic_metadata,
)


def summarise(tl, np_pairs, paths, cfg):
    n_tl = tl["timeline_id"].nunique() if len(tl) else 0
    print("\ntimelines")
    print(f"  built                 {n_tl}")
    print(f"  distinct patients     {tl['subject_id'].nunique() if len(tl) else 0}")
    if n_tl:
        print("  by split              " + ", ".join(
            f"{k}={v // cfg.length}" for k, v in tl['split'].value_counts().items()))
        gaps = tl["delta_tau_days"].dropna()
        print(f"  intervals             {len(gaps)}")
        print(f"    median gap          {gaps.median():.1f} d")
        print(f"    IQR                 {gaps.quantile(.25):.1f} - {gaps.quantile(.75):.1f} d")
        print(f"    min / max           {gaps.min():.2f} d / {gaps.max():.0f} d")
        print(f"    under 2 days        {(gaps < 2).mean():.1%}")
        print(f"    over 90 days        {(gaps > 90).mean():.1%}")
        print("  view mix              " + ", ".join(
            f"{k}={v:.1%}" for k, v in tl['ViewPosition'].value_counts(normalize=True).items()))

    print("\nnull-change pairs (weak supervision)")
    print(f"  pairs within 24h      {len(np_pairs)}")
    if len(np_pairs):
        print(f"  same calendar day     {np_pairs['same_day'].mean():.1%}")
        print(f"  projection changes    {np_pairs['projection_change'].mean():.1%}"
              "   <- what the reliability gate exists to catch")
        print(f"  median hours apart    {np_pairs['hours_apart'].median():.1f}")

    print(f"\nmanifest                {len(paths)} unique images "
          f"(~{len(paths) * 1.48 / 1024:.1f} GB raw, ~{len(paths) * 0.12 / 1024:.1f} GB at 512px)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata")
    ap.add_argument("--split")
    ap.add_argument("--out", default="cohort")
    ap.add_argument("--length", type=int, default=5)
    ap.add_argument("--overlap", action="store_true")
    ap.add_argument("--min-gap-days", type=float, default=0.0)
    ap.add_argument("--max-gap-days", type=float, default=None)
    a = ap.parse_args()

    cfg = CohortConfig(length=a.length, overlap=a.overlap,
                       min_gap_days=a.min_gap_days, max_gap_days=a.max_gap_days)

    if a.metadata:
        df = load_metadata(a.metadata, a.split)
        src = a.metadata
    else:
        raw = synthetic_metadata()
        tmp = pathlib.Path(a.out) / "_synthetic_metadata.csv"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(tmp, index=False)
        df = load_metadata(str(tmp))
        src = "SYNTHETIC STAND-IN (no --metadata given)"

    print("=" * 74)
    print("COHORT BUILDER")
    print(f"source: {src}")
    print(f"config: length={cfg.length} overlap={cfg.overlap} "
          f"min_gap={cfg.min_gap_days} max_gap={cfg.max_gap_days}")
    print("=" * 74)
    print(f"\nfrontal images after filtering: {len(df)}")

    tl = build_timelines(df, cfg)
    pairs = build_null_pairs(df, cfg)
    paths = manifest(tl, pairs)

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tl.to_csv(out / "timelines.csv", index=False)
    pairs.to_csv(out / "null_pairs.csv", index=False)
    (out / "manifest.txt").write_text("\n".join(paths) + "\n")

    summarise(tl, pairs, paths, cfg)
    print(f"\nwritten to {out}/  (timelines.csv, null_pairs.csv, manifest.txt)")


if __name__ == "__main__":
    main()
