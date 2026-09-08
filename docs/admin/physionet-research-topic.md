# PhysioNet credentialing — "Research Topic" field

Paste-ready text for the PhysioNet credentialed access application.
Adjust the affiliation sentence if applying as a student or under a supervisor.

---

## Full version (~300 words)

I am conducting research on longitudinal reasoning in chest radiography: modelling how radiographic
findings evolve across a patient's full sequence of visits, rather than comparing only a current
image against a single prior study. The intended output is a methodological paper submitted to
MICCAI 2027.

I plan to use the following datasets:

- **MIMIC-CXR-JPG v2.1.0** — chest radiograph images, together with the DICOM-derived metadata
  (ViewPosition, StudyDate, StudyTime) required to construct multi-visit patient timelines and to
  characterise acquisition differences between visits.
- **Chest ImaGenome v1.0.0** — the localized comparison relations (improved / worsened / no change)
  between anatomical regions across sequential exams, used as training supervision for a pairwise
  change classifier and to estimate disease transition dynamics as a function of the interval
  between studies.
- **MS-CXR-T v1.0.0** — expert-annotated disease progression labels, used as a held-out evaluation
  set to validate the change classifier against published benchmarks.
- **MIMIC-Ext-CXR-QBA v1.0.0** — structured question–answer annotations, used to construct training
  and validation splits for multi-visit reasoning tasks.

Methodologically, I will train a compact image comparator that classifies how a given finding
changes between two visits, then combine its outputs with a probabilistic temporal model to infer a
globally consistent disease trajectory across the visit sequence. I will additionally use pairs of
radiographs acquired on the same day to calibrate how much apparent change is attributable to
differences in patient positioning and acquisition technique rather than to genuine disease change.

All work is retrospective and computational, carried out on local hardware. I will not attempt to
identify any individual, will not link these data to any external dataset, and will not
redistribute the data or share my credentials. I will comply fully with the PhysioNet Credentialed
Health Data Use Agreement. Only aggregate results, model performance metrics, and de-identified
qualitative examples permitted under the DUA will appear in any publication.

---

## Condensed version (~120 words, if the field has a length limit)

I am researching longitudinal reasoning in chest radiography — modelling how findings evolve across
a patient's full sequence of visits rather than comparing only against a single prior study — for a
methods paper targeting MICCAI 2027.

I will use MIMIC-CXR-JPG v2.1.0 for images and acquisition metadata (ViewPosition, StudyDate) to
build multi-visit timelines; Chest ImaGenome v1.0.0 for localized improved/worsened/no-change
relations between sequential exams as training supervision; MS-CXR-T v1.0.0 as an expert-labelled
held-out evaluation set; and MIMIC-Ext-CXR-QBA v1.0.0 to construct training and validation splits.

Work is retrospective and computational. I will not attempt re-identification, will not link to
external datasets, will not redistribute data or share credentials, and will comply fully with the
PhysioNet Credentialed Health Data Use Agreement.

---

## Short version (~90 words) — recommended

I am researching longitudinal reasoning in chest radiography: modelling how findings evolve across
a patient's full sequence of visits rather than against a single prior study, for a methods paper
targeting MICCAI 2027.

I will use MIMIC-CXR-JPG v2.1.0 (images and acquisition metadata to build multi-visit timelines),
Chest ImaGenome v1.0.0 (improved/worsened/no-change relations between sequential exams as training
supervision), MS-CXR-T v1.0.0 (expert progression labels for held-out evaluation), and
MIMIC-Ext-CXR-QBA v1.0.0 (training and validation splits).

Work is retrospective and computational. I will not attempt re-identification, link to external
datasets, redistribute data, or share credentials, and will comply fully with the PhysioNet
Credentialed Health Data Use Agreement.

---

## Notes

- **Not for a class**, so no course name or number is included. If any part of this is coursework,
  add the course title and number — the form asks for it explicitly.
- If applying as a student, name your institution and supervisor. PhysioNet asks for a reference
  separately, and consistency between the two speeds up review.
- Apply the same text across all four datasets; each has its own DUA to sign, but the stated
  research purpose should match.
