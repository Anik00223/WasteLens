# WasteLens — Dataset Research (Pre-Training)

> Status: research only — **`src/train.py` has NOT been started** (awaiting confirmation).
> All counts below were verified against primary sources (dataset READMEs / GitHub mirror
> file listings), not blog posts. Sources listed per dataset.

## 1. Option A — TrashNet (Stanford, 6 classes)

**Source:** https://github.com/garythung/trashnet (README counts, quoted verbatim)
**Total: 2,527 images** · 512×384 JPEG · objects on a white posterboard, sunlight/room
lighting, iPhone cameras (7 Plus / 5S / SE).

| Class     | Images | % of total |
|-----------|-------:|-----------:|
| glass     |    501 |      19.8% |
| paper     |    594 |      23.5% |
| cardboard |    403 |      15.9% |
| plastic   |    482 |      19.1% |
| metal     |    410 |      16.2% |
| trash     |    137 |       5.4% |

**Class balance:** the `trash` class is significantly underrepresented — 137 images,
roughly **3–4× smaller** than every other class (~400–600).

**Mapped onto our 4-category taxonomy:**

| Taxonomy bin    | Images | Notes |
|-----------------|-------:|-------|
| Recyclable      |  2,390 | cardboard + glass + metal + paper + plastic (94.6%) |
| Organic         |      0 | **not covered at all** |
| Hazardous       |      0 | **not covered at all** |
| General trash   |    137 | the underrepresented class |

**Verdict:** 2 of our 4 taxonomy bins (organic, hazardous) have **zero images** in
TrashNet. It cannot train a 4-way classifier on its own.

## 2. Option B — Kaggle "Garbage Classification" (12 classes, ~4× larger)

**Source:** https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification
**Total: 15,515 images** (cross-confirmed by two independent mirrors:
https://github.com/nurulasyrifah/A-Convolutional-Neural-Network-for-Garbage-Classification-12-Classes
— per-class counts verified file-by-file — and
https://github.com/aesopdev/garbage_classification_model_smam, "trained on 15,515 images").

| Class        | Images | % of total |
|--------------|-------:|-----------:|
| clothes      |  5,325 |      34.3% |
| shoes        |  1,977 |      12.7% |
| paper        |  1,050 |       6.8% |
| biological   |    985 |       6.3% |
| battery      |    945 |       6.1% |
| cardboard    |    891 |       5.7% |
| plastic      |    865 |       5.6% |
| metal        |    769 |       5.0% |
| white-glass  |    775 |       5.0% |
| trash        |    697 |       4.5% |
| green-glass  |    629 |       4.1% |
| brown-glass  |    607 |       3.9% |

**Class balance:** at the 12-class level, `brown-glass` (607) / `green-glass` (629) /
`trash` (697) are the smallest classes; `clothes` (5,325) heavily dominates — ~8.8× the
smallest class.

**Mapped onto our 4-category taxonomy (glass variants merged, clothes/shoes handled two
ways):**

| Taxonomy bin    | v1: drop clothes/shoes | v2: fold clothes/shoes → general trash |
|-----------------|-----------------------:|---------------------------------------:|
| Recyclable      |  5,586 (68.0%) |  5,586 (36.0%) |
| Organic         |    985 (12.0%) |    985 ( 6.3%) |
| Hazardous       |    945 (11.5%) |    945 ( 6.1%) |
| General trash   |    697 ( 8.5%) |  7,999 (51.5%) |
| **Total used**  | **8,213** | **15,515** |

- v1 (drop): cleanest semantics, but wastes 7,302 collected images and leaves an ~**8:1**
  max/min imbalance (recyclable 5,586 vs general trash 697).
- v2 (fold): uses everything, but clothes/shoes are textile items whose "general trash"
  label is arguable (textiles often have separate recycling streams) — injects label noise.

**Verdict:** the **only single dataset that covers all four of our taxonomy bins**,
including hazardous (battery) and organic (biological).

## 3. Option C — RealWaste (Univ. of Wollongong, 9 classes)

**Source:** https://github.com/sam-single/realwaste (official authors' mirror; cites
"RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep
Learning", MDPI Information 14(12):633). **License: CC BY-NC-SA 4.0** (non-commercial —
fine for a college project, must attribute).
**Total: 4,752 images** · photos of waste in authentic state at the Whyte's Gully
recovery facility — real-world conditions, closest match to our camera-captured demo.

| Class              | Images | % of total |
|--------------------|-------:|-----------:|
| plastic            |    921 |      19.4% |
| metal              |    790 |      16.6% |
| paper              |    500 |      10.5% |
| miscellaneous trash|    495 |      10.4% |
| vegetation         |    436 |       9.2% |
| cardboard          |    461 |       9.7% |
| glass              |    420 |       8.8% |
| food organics      |    411 |       8.6% |
| textile trash      |    318 |       6.7% |

**Class balance:** fairly balanced at 9-class level (318–921).

**Mapped onto our 4-category taxonomy:**

| Taxonomy bin    | Images | Notes |
|-----------------|-------:|-------|
| Recyclable      |  3,092 | cardboard + glass + metal + paper + plastic (65.1%) |
| Organic         |    847 | food organics + vegetation (17.8%) |
| General trash   |    813 | misc trash + textile trash (17.1%) |
| Hazardous       |      0 | **no battery/hazardous class** |

**Verdict:** best real-world image conditions, no hazardous images.

## 4. Option D — Kaggle "Waste Classification Data" (2 classes)

**Source:** https://www.kaggle.com/techsash/waste-classification-data (counts from the
authors' repo, https://github.com/techSash/Waste-classification)
**Total: 25,077 images** — train 22,564 (12,565 organic / 9,999 recyclable) + test 2,513
(1,401 organic / 1,112 recyclable).

| Taxonomy bin    | Images | Notes |
|-----------------|-------:|-------|
| Organic         | 13,966 | 55.7% |
| Recyclable      | 11,111 | 44.3% |
| Hazardous       |      0 | not covered |
| General trash   |      0 | not covered (recyclable/organic only) |

**Verdict:** largest by volume, but only 2 taxonomy bins — cannot train a 4-way
classifier on its own.

## 5. Comparison Summary (mapped to our 4 bins)

| Option             | Total  | Recyclable | Organic | Hazardous | General trash               | Covers all 4 bins? |
|--------------------|-------:|-----------:|--------:|----------:|----------------------------:|:------------------:|
| A. TrashNet        |  2,527 | 2,390      | 0       | 0         | 137                         | **No**             |
| B. Kaggle 12-class | 15,515 | 5,586      | 985     | 945       | 697 (drop) / 7,999 (fold)   | **Yes**            |
| C. RealWaste       |  4,752 | 3,092      | 847     | 0         | 813                         | **No**             |
| D. Kaggle O/R      | 25,077 | 11,111     | 13,966  | 0         | 0                           | **No**             |

**Underrepresentation findings:**
- TrashNet: `trash` 137 (5.4%) — significantly underrepresented; organic/hazardous absent.
- Kaggle 12-class: after mapping with drop, `general trash` 697 is smallest (~8:1 vs
  recyclable); `clothes` 5,325 dominates the raw 12-class distribution.
- RealWaste: balanced 9-class (318–921), but hazardous absent.
- O/R: only 2 bins; hazardous/general absent.

## 6. Recommendation

**Primary: Option B — Kaggle "Garbage Classification" (12 classes).**

Why:
1. **It is the only single dataset that covers all four of our taxonomy bins**,
   including the hazardous (battery: 945) and organic (biological: 985) classes that
   TrashNet and RealWaste completely lack — and hazardous is the class most at risk in
   our per-class evaluation criterion.
2. **15,515 images ≫ TrashNet's 2,527**, giving roughly 3,500–4,500 usable images per
   taxonomy bin after mapping — enough for transfer learning with a frozen backbone.
3. Counts are independently verified (file-by-file) and the dataset is widely used and
   easily re-downloadable from Kaggle.

Proposed mapping (v1): cardboard/paper/plastic/metal/{brown,green,white}-glass →
**recyclable**; biological → **organic**; battery → **hazardous**; trash → **general
trash**; **drop clothes & shoes** (7,302 images, arguable labels — folding them into
general trash is a v2 experiment if we need more volume).

Known risks with this choice (to address in training with class weights + stratified
split): ~8:1 max/min class imbalance; `general trash` is smallest AND most
heterogeneous — expected to be the weak-performing class in the per-class report.

**Optional secondary data (only if time permits):** merge TrashNet (+2,527 images, adds
137 real trash images) and/or RealWaste (best real-condition photos — improves demo
robustness to camera-captured images vs TrashNet's studio-style shots). Neither fixes
the hazardous gap; if battery/hazardous per-class recall is poor, a small web-scraped
hazardous set would be the fallback — to be discussed separately.


