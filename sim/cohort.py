"""Five-visit cohort builder over MIMIC-CXR-JPG metadata.

Runs entirely off `mimic-cxr-2.0.0-metadata.csv.gz` (a few MB) plus the official split file.
No images are needed to build the cohort - the manifest it emits is what tells you which
images to download, which is the point of the disk strategy in docs/data.md.

It produces the three things the method needs:

  timelines    ordered runs of `length` frontal studies per patient, with the true inter-visit
               interval in days. Delta-tau is what the CTMC prior conditions on (proposal 3.7),
               and MIMIC's date shift is per-patient and constant, so within-patient
               differences are exact even though absolute dates are not real.

  null pairs   same-patient, same-day study pairs. These are WEAK null-change supervision for
               the reliability gate (proposal 3.6) - weak, not gold, because pneumothorax,
               flash edema and post-drainage effusion genuinely change within hours.

  manifest     the unique JPG paths referenced, for a targeted download.

Patients never cross splits: every timeline from a subject inherits that subject's split.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FRONTAL = ("PA", "AP")
REQUIRED = ["subject_id", "study_id", "dicom_id", "ViewPosition", "StudyDate", "StudyTime"]


def jpg_path(subject_id: int, study_id: int, dicom_id: str) -> str:
    """MIMIC-CXR-JPG layout: files/p10/p10000032/s50414267/<dicom_id>.jpg"""
    s = str(subject_id)
    return f"files/p{s[:2]}/p{s}/s{study_id}/{dicom_id}.jpg"


def _timestamp(date: pd.Series, time: pd.Series) -> pd.Series:
    """Absolute time in days. StudyTime is HHMMSS.SS, so seconds need base-60 unpacking.

    The date part goes through `.dt.total_seconds()` rather than `.astype("int64")`. Pandas 2+
    infers datetime resolution, so the integer view is not reliably nanoseconds; dividing by a
    hardcoded 86.4e12 silently collapsed every date to nearly zero and left only the
    within-day fraction. That would have corrupted every delta-tau - the CTMC's whole input -
    without raising anything. See test_cohort_timestamps.
    """
    d = pd.to_datetime(date.astype("int64").astype(str), format="%Y%m%d")
    days = (d - pd.Timestamp("1970-01-01")).dt.total_seconds() / 86400.0
    t = pd.to_numeric(time, errors="coerce").fillna(0.0)
    hh = np.floor(t / 10000.0)
    mm = np.floor((t - hh * 10000.0) / 100.0)
    ss = t - hh * 10000.0 - mm * 100.0
    return days + (hh * 3600.0 + mm * 60.0 + ss) / 86400.0


@dataclass
class CohortConfig:
    length: int = 5
    overlap: bool = False          # non-overlapping windows avoid near-duplicate timelines
    min_gap_days: float = 0.0      # drop a run if any interval is shorter than this
    max_gap_days: float | None = None
    prefer_view: str = "PA"        # one frontal image per study; PA is less magnified than AP


def load_metadata(path: str, split_path: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"metadata is missing required columns: {missing}")
    df = df[df["ViewPosition"].isin(FRONTAL)].copy()
    df["t_days"] = _timestamp(df["StudyDate"], df["StudyTime"])
    if split_path:
        sp = pd.read_csv(split_path)[["subject_id", "split"]].drop_duplicates("subject_id")
        df = df.merge(sp, on="subject_id", how="left")
        df["split"] = df["split"].fillna("unknown")
    else:
        df["split"] = "unknown"
    return df


def one_image_per_study(df: pd.DataFrame, cfg: CohortConfig) -> pd.DataFrame:
    """Collapse each study to a single frontal image, preferring the configured view."""
    d = df.copy()
    d["_pref"] = (d["ViewPosition"] != cfg.prefer_view).astype(int)
    d = d.sort_values(["subject_id", "study_id", "_pref", "dicom_id"])
    return d.drop_duplicates(["subject_id", "study_id"], keep="first").drop(columns="_pref")


def build_timelines(df: pd.DataFrame, cfg: CohortConfig) -> pd.DataFrame:
    """Ordered runs of `cfg.length` studies per patient, with per-interval delta-tau."""
    studies = one_image_per_study(df, cfg).sort_values(["subject_id", "t_days", "study_id"])
    rows = []
    tl_id = 0
    step = 1 if cfg.overlap else cfg.length

    for subject, g in studies.groupby("subject_id", sort=True):
        g = g.reset_index(drop=True)
        for start in range(0, len(g) - cfg.length + 1, step):
            w = g.iloc[start:start + cfg.length]
            gaps = np.diff(w["t_days"].to_numpy())
            if cfg.min_gap_days and (gaps < cfg.min_gap_days).any():
                continue
            if cfg.max_gap_days is not None and (gaps > cfg.max_gap_days).any():
                continue
            for pos, (_, r) in enumerate(w.iterrows()):
                rows.append({
                    "timeline_id": tl_id,
                    "subject_id": subject,
                    "split": r["split"],
                    "position": pos + 1,
                    "study_id": r["study_id"],
                    "dicom_id": r["dicom_id"],
                    "path": jpg_path(subject, r["study_id"], r["dicom_id"]),
                    "ViewPosition": r["ViewPosition"],
                    "StudyDate": r["StudyDate"],
                    "days_from_first": r["t_days"] - w["t_days"].iloc[0],
                    # delta_tau_days is the gap to the NEXT visit; NaN at the last position
                    "delta_tau_days": gaps[pos] if pos < cfg.length - 1 else np.nan,
                })
            tl_id += 1

    cols = ["timeline_id", "subject_id", "split", "position", "study_id", "dicom_id",
            "path", "ViewPosition", "StudyDate", "days_from_first", "delta_tau_days"]
    return pd.DataFrame(rows, columns=cols)


def build_null_pairs(df: pd.DataFrame, cfg: CohortConfig,
                     max_hours: float = 24.0) -> pd.DataFrame:
    """Same-patient study pairs acquired within `max_hours`.

    WEAK null-change supervision (proposal 3.6). True disease change is small but not zero,
    so `projection_change` is surfaced: pairs that swap AP<->PA are the ones where apparent
    change is most likely to be acquisition rather than disease, and they are also the pairs
    the reliability gate exists to catch.
    """
    studies = one_image_per_study(df, cfg).sort_values(["subject_id", "t_days"])
    rows = []
    for subject, g in studies.groupby("subject_id", sort=True):
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            for j in range(i + 1, len(g)):
                hours = (g.at[j, "t_days"] - g.at[i, "t_days"]) * 24.0
                if hours > max_hours:
                    break
                rows.append({
                    "subject_id": subject,
                    "split": g.at[i, "split"],
                    "study_a": g.at[i, "study_id"], "study_b": g.at[j, "study_id"],
                    "path_a": jpg_path(subject, g.at[i, "study_id"], g.at[i, "dicom_id"]),
                    "path_b": jpg_path(subject, g.at[j, "study_id"], g.at[j, "dicom_id"]),
                    "view_a": g.at[i, "ViewPosition"], "view_b": g.at[j, "ViewPosition"],
                    "hours_apart": hours,
                    "same_day": bool(g.at[i, "StudyDate"] == g.at[j, "StudyDate"]),
                    "projection_change": g.at[i, "ViewPosition"] != g.at[j, "ViewPosition"],
                })
    cols = ["subject_id", "split", "study_a", "study_b", "path_a", "path_b",
            "view_a", "view_b", "hours_apart", "same_day", "projection_change"]
    return pd.DataFrame(rows, columns=cols)


def manifest(*frames: pd.DataFrame) -> list[str]:
    """Unique JPG paths across the given frames, for a targeted download."""
    paths: set[str] = set()
    for f in frames:
        for col in ("path", "path_a", "path_b"):
            if col in f.columns:
                paths.update(f[col].dropna().tolist())
    return sorted(paths)


# ---------------------------------------------------------------- synthetic stand-in


def synthetic_metadata(n_subjects: int = 400, seed: int = 0) -> pd.DataFrame:
    """A stand-in with MIMIC-CXR-JPG's schema, so the builder is testable before access.

    Deliberately includes the awkward cases: lateral-only studies, missing ViewPosition,
    same-day repeats, multiple images per study, and intervals from hours to years.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_subjects):
        subject = 10_000_000 + k
        n_studies = int(rng.integers(1, 12))
        day = float(rng.integers(0, 3000))
        for _ in range(n_studies):
            study = int(rng.integers(50_000_000, 59_999_999))
            for _ in range(int(rng.integers(1, 3))):     # 1-2 images per study
                view = rng.choice(["PA", "AP", "LATERAL", None], p=[0.30, 0.50, 0.18, 0.02])
                rows.append({
                    "dicom_id": f"{rng.integers(16**8):08x}-{rng.integers(16**8):08x}",
                    "subject_id": subject,
                    "study_id": study,
                    "ViewPosition": view,
                    "StudyDate": int((pd.Timestamp("2100-01-01")
                                      + pd.Timedelta(days=day)).strftime("%Y%m%d")),
                    "StudyTime": float(rng.integers(0, 24) * 10000
                                       + rng.integers(0, 60) * 100 + rng.integers(0, 60)),
                    "Rows": 2544, "Columns": 3056,
                    "PerformedProcedureStepDescription": "CHEST (PA AND LAT)",
                })
            # next study: usually a new day, sometimes a same-day repeat
            day += 0.0 if rng.random() < 0.18 else float(np.exp(rng.uniform(0, 6.2)))
    return pd.DataFrame(rows)
