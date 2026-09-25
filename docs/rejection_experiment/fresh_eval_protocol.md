# WasteLens — Iteration 9: PRE-REGISTERED fresh real-world evaluation protocol

Written **before** any fresh image was downloaded and before any fresh metric
was computed. `src/build_fresh_set.py` implements exactly this procedure; the
downloaded images never enter the repository (only
`docs/rejection_experiment/fresh_set_manifest.json` — source URLs, labels,
licences, SHA-256 — is committed).

## 1. Freshness claim (what "fresh" means here)

* Source: **Wikimedia Commons** (per-file page, licence and author recorded).
* The project's existing data comes from: Kaggle `mostafaabla/
  garbage-classification` v1 (training/validation/test + clothes/shoes),
  CIFAR-10 (non-waste), and collages constructed from those pools. **None of
  the fresh images come from those sources.**
* Verifiable: the builder computes SHA-256 of every downloaded image and
  compares it against SHA-256 of **every image file under the training corpus
  root** (Kaggle collection) and the CIFAR PNG cache; the manifest records
  `dedup_matches` (expected: empty). It also records every file's Commons page
  URL + licence, so provenance is auditable per image.
* The 5 synthetic probes are generated in code
  (`src/validate_realworld.py::ood_probes`, seed 7) and are reported as their
  own category; they were never trained on.

## 2. Deterministic selection procedure

1. For each category below: `action=query&list=categorymembers&cmtype=file`,
   sorted by file title (Commons sortkey order), `cmlimit=500`.
2. Keep only files whose title ends in `.jpg/.jpeg/.png` **and** whose
   `imageinfo.mediatype == BITMAP` (excludes audio/video/vector).
3. Pre-registered title exclusions (case-insensitive substring): `diagram`,
   `map`, `logo`, `icon`, `chart`, `poster`, `stamp`, `coin`, `engraving`,
   `lithograph`, `drawing`, `painting`, `sketch`, `screenshot`, `book`,
   `magazine`, `manuscript`, `coat of arms`, `graph`, `blueprint`, `svg`.
4. Geometry filter: at least 400 px on the long side of the 1280 px
   thumbnail, aspect ratio ≤ 4.0.
5. Take the first **N** survivors per category in title order (no manual
   selection, no model involvement).
6. Download the Commons thumbnail at `iiurlwidth=1280` (Wikimedia re-encoded
   JPEG) from `upload.wikimedia.org`; record sha256, size, source page,
   licence, author, original dimensions.

## 3. Pre-registered categories, counts and labels

| group | Commons category | N | label |
|---|---|---:|---|
| supported-organic | Category:Compost | 6 | organic |
| supported-organic | Category:Food waste | 6 | organic |
| supported-organic | Category:Compost bins | 4 | organic |
| supported-recyclable | Category:Cardboard boxes | 6 | recyclable |
| supported-recyclable | Category:Aluminium cans | 5 | recyclable |
| supported-recyclable | Category:Plastic bottles | 5 | recyclable |
| supported-recyclable | Category:Broken glass | 5 | recyclable (glass → recyclable, as in training) |
| supported-hazardous | Category:Lead-acid batteries | 5 | hazardous |
| supported-hazardous | Category:Button cells | 5 | hazardous |
| supported-hazardous | Category:Fluorescent lamps | 5 | hazardous |
| supported-general | Category:Litter | 6 | general trash |
| supported-general | Category:Paper towels | 5 | general trash |
| **supported total** | | **63** | |
| ambiguous | Category:Pizza boxes | 5 | ambiguous (recyclable vs contaminated) |
| ambiguous | Category:Cigarette butts | 5 | ambiguous (general vs hazardous) |
| **ambiguous total** | | **10** | |
| ood-objects | Category:Bicycles | 5 | unsupported |
| ood-objects | Category:Toy robots | 5 | unsupported |
| ood-objects | Category:Computer keyboards | 5 | unsupported |
| ood-clothes | Category:Shirts | 8 | unsupported |
| ood-shoes | Category:Shoes | 8 | unsupported |
| ood-scenes | Category:Parks | 6 | unsupported (scene) |
| ood-scenes | Category:Streets | 6 | unsupported (scene) |
| **ood total** | | **43** | |
| multi-object | Category:Beach litter | 6 | multi-object (real photos) |
| multi-object | Category:Flea markets | 6 | multi-object (real photos) |
| multi-object | Category:Clutter | 4 | multi-object (real photos) |
| **multi-object total** | | **16** | |
| synthetic | code-generated probes | 5 | unsupported (reported separately) |
| **grand total** | | **137** | |

If a category returns fewer than N usable files, the builder takes what exists
and records the shortfall; categories are **not** substituted after the fact.

## 4. Locked evaluation settings

* model: shipped `models/checkpoints/wastelens_rej_shipped_best.keras`
  (Variant C, seed 42, epoch 10) — never retrained in this iteration.
* **production rejection threshold: `0.0702` (unchanged, no tuning).**
* uncertainty rule unchanged: top < 0.60 or margin < 0.50 → uncertain;
  rejection has priority (`unsupported → uncertain → supported`).
* preprocessing: **primary = the shipped Python path**
  (`tf.io.decode_image → tf.image.resize → mobilenet preprocess`, exactly the
  path behind every shipped metric); **secondary = browser-equivalent**
  (Pillow decode + tfjs `y*in/out` bilinear), reported for comparison only —
  never used to decide anything in this iteration.
* four-bin order unchanged: recyclable, organic, hazardous, general trash.

## 5. Metrics (pre-registered)

* classification (supported 63 only): accuracy, macro-F1, per-bin P/R/F1 and
  confusion matrix.
* rejection: AUROC (supported = negative; OOD 43 + multi-object 16 = positive),
  FRR at 0.0702, OOD detection at 0.0702, reject precision/recall at 0.0702.
* decision states: counts of supported / uncertain / unsupported **per
  category group** (supported, ambiguous, ood-*, multi-object, synthetic).
* multi-object (LOOP 9): detection rate at 0.0702 next to the supported
  false-rejection rate, reject-score distribution (min/median/max/mean) and
  top-bin distribution. No claim beyond the 16 sampled photos.
* ambiguous group: reported as states + predicted-bin distribution; **excluded
  from the 4-bin accuracy** because its ground-truth bin is definitionally
  debatable.
* operating points (LOOP 10): thresholds re-derived from the shipped model's
  supported-**validation** scores at 1%, 2% and 5% FRR (production path),
  then applied to the shipped test set and to the fresh set. Measurement only
  — production stays at 0.0702.

## 6. What must not happen

No threshold change. No category substitution after seeing results. No claim
of "fresh" without the recorded provenance + dedup evidence. No training, no
re-calibration, no change to `web/`.

