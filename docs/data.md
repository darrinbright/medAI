# PhysioNet data requirements

All five items sit behind the **same one-time credentialing** (PhysioNet account + CITI "Data or
Specimens Only Research" course). Once credentialed you sign each project's DUA separately, which
is near-instant. **Apply for all of them in one sitting.**

Register: https://physionet.org/register/ → complete CITI → sign each DUA below.

---

## Required

### 1. MIMIC-CXR-JPG v2.1.0 — images + acquisition metadata
https://physionet.org/content/mimic-cxr-jpg/2.1.0/

377,110 JPG images, 227,827 studies, 65,379 patients (BIDMC 2011–2016). **558 GB in full** — see
§ Disk strategy below, do not download it all.

Beyond the images, this is the source of every acquisition covariate that Stage 2 needs:

| File | Supplies |
|---|---|
| `mimic-cxr-2.0.0-metadata.csv.gz` | `ViewPosition` (AP/PA/LATERAL), `StudyDate`, `StudyTime`, `Rows`, `Columns`, `PerformedProcedureStepDescription` |
| `mimic-cxr-2.0.0-split.csv.gz` | official train/val/test split — respect it to stay comparable |
| `mimic-cxr-2.0.0-chexpert.csv.gz` / `-negbio.csv.gz` | finding labels, useful for the unary severity head |

`StudyDate` + `StudyTime` are what make **Stage 3 (Δτ-conditioned CTMC priors)** possible, and
`subject_id`/`study_id`/`StudyDate` are what let you mine **same-day repeat pairs for the Stage 2
null-change trick**. Both novel contributions are unlocked by this one CSV — grab it first, it is
a few MB and you can prototype the cohort logic before any image lands.

### 2. Chest ImaGenome v1.0.0 — the primary training signal
https://physionet.org/content/chest-imagenome/1.0.0/

242,072 scene graphs over 217,013 studies, 29 anatomical regions with bounding boxes, and —
critically — **over 670,000 localized comparison relations (improved / worsened / no change)
between anatomical locations across sequential exams**.

That is exactly the supervision the Stage 1 comparator needs, at a scale no other public resource
offers. It also provides the progression sequences for MLE-estimating the per-finding CTMC
generator `Q_f`. Ships as `silver_dataset/` (auto-generated) and `gold_dataset/` (500 patients,
manually validated — use as a clean eval slice and to measure silver label noise).

### 3. MIMIC-Ext-CXR-QBA v1.0.0 — MI-CXR's substrate
https://physionet.org/content/mimic-ext-cxr-qba/1.0.0/

42M QA pairs derived from MIMIC-CXR with hierarchical answers, bounding boxes and structured
region/finding tags (Müller, Jungmann, Kaissis & Rueckert, 2025).

The MI-CXR README names this and MIMIC-CXR-JPG as its two dependencies. Required if you want to
regenerate, extend, or build a *training* split for MI-CXR — the public GitHub release is
test-only, so you cannot train on it without this.

### 4. MS-CXR-T v1.0.0 — expert progression labels
https://physionet.org/content/ms-cxr-t/1.0.0/

CSV annotations only, a few MB. 1,326 expert-labelled image pairs across 5 findings
({improving, stable, worsening}) plus 361 temporal sentence-similarity pairs. Images resolve
against your MIMIC-CXR-JPG download.

Two uses: clean expert-labelled supervision to counterbalance Chest ImaGenome's silver noise, and
a secondary benchmark where BioViL-T and CoCa-CXR (65.0%) give published comparison points for the
Stage 1 comparator in isolation.

---

## Optional

### 5. MIMIC-CXR v2.1.0 — reports archive only
https://physionet.org/content/mimic-cxr/2.1.0/

**Download only the reports archive. Skip the DICOMs — that project is 4.6 TB.**

Free-text reports are useful for verifying null-change pairs (explicit "no interval change"
phrasing), for qualitative error analysis, and for the option→predicate parser sanity checks.
Not on the critical path.

---

## Not from PhysioNet (free, no credentialing)

| Resource | Where |
|---|---|
| MI-CXR test set (`micxr_test.jsonl`, 5,311 items) | https://github.com/AIDASLab/MI-CXR |
| GRCD cleaned Chest ImaGenome pairs (40,250) | GRCD repo (arXiv 2607.02719) |

---

## Disk strategy — the actual constraint

558 GB of JPGs is the binding practical problem on a desktop, and you do not need it.

**Do not mirror the archive. Build a manifest, fetch only what it names, resize on ingest.**

1. Download the metadata CSVs first (a few MB, no images).
2. Resolve the exact `dicom_id` set you need:
   - MI-CXR: the image paths in `micxr_test.jsonl`. 5,311 items × 5 images, but timelines are
     shared across TEL/ICR/GTS, so unique images are far fewer than the 26,555 slots.
   - Comparator training: sample studies from Chest ImaGenome comparison relations.
   - Null-change pairs: same `subject_id` + same `StudyDate`, distinct `study_id`.
3. Fetch by path with credentialed wget:

```bash
wget -r -N -c -np --user <USERNAME> --ask-password \
  https://physionet.org/files/mimic-cxr-jpg/2.1.0/files/p10/p10000032/s50414267/
# or batch from a manifest:
wget -N -c -i manifest.txt --user <USERNAME> --ask-password
```

PhysioNet also mirrors MIMIC-CXR-JPG to **GCS and S3 (requester-pays)**, which is better suited to
selective fetch than recursive wget if you have cloud credit.

4. **Resize-on-ingest, then delete the original.** Images average ~1.48 MB (558 GB / 377,110). At
   512×512 JPEG q90 they drop to roughly 80–150 KB — a ~10× reduction, and 512px is the input
   resolution the Stage 1 comparator trains at anyway.

Rough budget with that pipeline:

| Slice | Images | Raw | After 512px resize |
|---|---|---|---|
| MI-CXR evaluation timelines | ~5–10k | ~7–15 GB | ~1–2 GB |
| Comparator training pool | ~100k | ~150 GB | ~10–15 GB |
| Null-change calibration pairs | ~20k | ~30 GB | ~2–3 GB |

Streaming download → resize → discard keeps peak disk in the tens of GB rather than hundreds.
Budget ~1 TB of free disk to be comfortable, but it is workable on far less.

---

## Sequencing against the critical path

Credentialing is the only true blocker, so order the work to hide it:

1. **Today** — register, start CITI, submit. Approval typically 3–14 days.
2. **While waiting** — build the synthetic simulation harness (proposal §8). It needs no data at
   all: implement the CTMC prior and forward–backward, then sweep achievable accuracy as a
   function of local comparator accuracy and calibration. This tells you whether the method can
   work *before* a single image downloads.
3. **On approval** — pull metadata CSVs and Chest ImaGenome first (small, and they determine the
   manifest). Start the image fetch in the background; it will run for a while.
