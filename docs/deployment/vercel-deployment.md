# WasteLens AI — Vercel Production Deployment Record

This document records the official production deployment of WasteLens AI to Vercel for academic evaluation and faculty submission.

---

## 1. Deployment Metadata

| Field | Value |
|---|---|
| **Platform** | Vercel |
| **Vercel Project Name** | `wastelens-final-submission` |
| **Vercel Project ID** | `prj_e0E4HFD55r4UPI4L2YK3KIVKFgXo` |
| **Account / Scope** | `anik00223s-projects` |
| **Permanent Production URL** | [https://wastelens-final-submission.vercel.app](https://wastelens-final-submission.vercel.app) |
| **Immutable Deployment URL** | [https://wastelens-final-submission-qmuuq8nv6-anik00223s-projects.vercel.app](https://wastelens-final-submission-qmuuq8nv6-anik00223s-projects.vercel.app) |
| **Vercel Inspector URL** | [https://vercel.com/anik00223s-projects/wastelens-final-submission/286umhzYPRHGsbL23rkMJ3UoFQ61](https://vercel.com/anik00223s-projects/wastelens-final-submission/286umhzYPRHGsbL23rkMJ3UoFQ61) |
| **Source Git Repository** | [https://github.com/Anik00223/WasteLens](https://github.com/Anik00223/WasteLens) |
| **Source Git Branch** | `fix/final-submission` |
| **Source Commit SHA** | `7b6cf575c028376703f45dd971c406e4f7453c20` |
| **Deployment Root Directory** | `web/` |
| **Framework Preset** | `Other` (Static Website) |
| **Build Command** | None |
| **Install Command** | None |
| **Output Directory** | `.` |
| **Deployment Timestamp** | 2026-11-08 23:46:43 IST |

---

## 2. Architecture & Preservation Constraints

In accordance with academic submission criteria:
1. **Model Weights & Graph:** MobileNetV2 dual-head architecture completely preserved from frozen checkpoints.
2. **OOD Rejection Gate:** Calibrated sigmoid threshold of `0.0702` preserved without alteration.
3. **Labels & Preprocessing:** 4 classification bins (`recyclable`, `organic`, `hazardous`, `general trash`), $[0, 1]$ normalization, and $[224, 224]$ bilinear resize remain intact.
4. **Default Mode:** Single Item classification remains the primary verified submission workflow.
5. **Multi-Region Mode:** Maintained as an experimental 8-window spatial heuristic with prominent academic disclosures.
6. **Production Fallback:** Existing baseline deployment on Antideploy (`https://wastelens.antideploy.app`) and branch `release/wastelens-final` (`ddd5626`) remain strictly unmodified.

---

## 3. Live Browser Verification Results

Automated headless Chrome CDP end-to-end verification executed against `https://wastelens-final-submission.vercel.app`:

| Verification Phase | Test Case / Fixture | Observed Result | Status |
|---|---|---|---|
| **Website Availability** | Root URL (`/`) | HTTP 200 OK, Content-Type: `text/html; charset=utf-8`, 89,460 bytes | PASS |
| **Model Topology** | `/model/model.json` | HTTP 200 OK, 215,194 bytes, TF.js graph definition verified | PASS |
| **Weight Shards** | `/model/group1-shard[1-3]of3.bin` | HTTP 200 OK across all 3 binary shards (10.35 MB total) | PASS |
| **Runtime Initialization** | Browser TF.js engine | TF.js 4.22.0 initialized, warm-up tensor executed, UI reads "Model ready" | PASS |
| **Single Item (In-Distribution)** | `supported-recyclable__Aluminium_cans__00.jpg` | Classified as **Recyclable** (Confidence: 100.0%, OOD Score: `0.0001` < `0.0702`) | PASS |
| **Single Item (OOD Rejection)** | `ood-scenes__Parks__00.jpg` | Rejected as **Unsupported** (Confidence: 97.6%, OOD Score: `0.9762` $\ge$ `0.0702`) | PASS |
| **Single Item (Ambiguous Fixture)** | `ambiguous__Pizza_boxes__00.jpg` | Classified as **Recyclable** (Confidence: 100.0%, OOD Score: `0.0000`) | PASS |
| **Multi-Region Mode** | `supported-general__Litter__01.jpg` | 4 Candidate regions localized with bounding boxes; experimental warning badge visible | PASS |
| **Academic Transparency** | Anchor `#audit` | 14-image empirical audit narrative and 5-dimension comparative table verified | PASS |
| **Mobile Responsiveness** | Viewport 390x844 (Mobile) | Drawer sidebar toggle, responsive dropzone, header visibility all functional | PASS |
| **Console & Network Health** | Browser DevTools log | **0 console errors**, 0 network errors, 0 JavaScript exceptions | PASS |

---

## 4. Submission Artifacts & Verification Proofs

- Visual screenshots captured from the live Vercel deployment:
  - Desktop Landing: `vercel_desktop_landing.png`
  - Single-Item Classification: `vercel_desktop_single_result.png`
  - OOD Rejection (Unsupported): `vercel_desktop_unsupported.png`
  - Multi-Region Analysis: `vercel_desktop_multi_result.png`
  - Mobile Viewport: `vercel_mobile_view.png`
- Test harness: `scratch/test_live_vercel.js`
