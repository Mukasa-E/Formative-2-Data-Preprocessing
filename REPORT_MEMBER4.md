# Member 4 Report — CLI App, System Simulation & Demo Video

**Formative 2 — Multimodal Data Preprocessing**  
**Role:** Member 4 — CLI Application, GitHub Repository, Report, Demo Video  
**System:** User Identity & Product Recommendation System

---

## 1. Overview

Member 4's responsibility is to integrate the artefacts produced by the other
team members into a working command-line system simulation that demonstrates
the full end-to-end flow:

1. **Face image** → Face Recognition Model (recognise a known team member)
2. **Voice sample** → Voiceprint Verification Model (approve an enrolled voice)
3. **Both pass** → Product Recommendation Model predicts the product category
4. **Any failure** → ACCESS DENIED (no recommendation served)

This implements the rubric's **System Simulation** criterion (4 pts): a full
authorised transaction plus an unauthorised-attempt demo, working in CLI.

---

## 2. Models Used

| Model | Purpose | Source | File |
|-------|---------|--------|------|
| Face Recognition | Recognise a team member from a facial image | Trained by Member 4 from `image_features.csv` | `models/face_model.joblib` |
| Voiceprint Verification | Verify an enrolled voice from an audio sample | Trained by Member 4 from `audio_features.csv` | `models/voice_model.joblib` |
| Product Recommendation | Predict product category for a customer | Trained by product team | `models/product_model.joblib` |

### Face Recognition Model
- **Algorithm:** Random Forest (`n_estimators=200`, `max_depth=8`, class-balanced)
- **Features:** 660-dim vector per image (intensity statistics, RGB histograms,
  LBP texture, HOG shape) — extracted by Member 2's image pipeline
- **Training data:** `image_processing/output/image_features.csv` (4 members:
  anglebert, mukasa, nina, samuel; neutral/smiling/surprised expressions;
  original + 4 augmentations each)
- **Performance:** accuracy = 0.9333, F1-weighted = 0.9323 (25% hold-out test)

### Voiceprint Verification Model
- **Algorithm:** Random Forest (`n_estimators=200`, `max_depth=8`, class-balanced)
- **Features:** MFCC means/stds (13 coefficients), spectral roll-off, RMS energy,
  spectral centroid, zero-crossing rate — extracted by Member 3's audio pipeline
- **Training data:** `data/audio/audio_features.csv` (4 members × 2 phrases ×
  original + 3 augmentations)
- **Performance:** accuracy = 1.0000, F1-weighted = 1.0000 (25% hold-out test)

### Product Recommendation Model
- **Algorithm:** Random Forest (selected over Logistic Regression and XGBoost by
  weighted F1 on the product team's evaluation)
- **Features:** 29 engineered features from `data/merged_dataset.csv`
  (purchase amount, rating, engagement, purchase interest, platform, sentiment,
  date features, customer-level aggregates, amount buckets)
- **Performance:** accuracy = 0.2632, F1-weighted = 0.2595 (25% hold-out test)
  — note: product category prediction is inherently noisy with 5 categories;
  the model still outperforms the majority-class baseline (0.2368).

---

## 3. CLI Application — `system_app.py`

### 3.1 System Flow

```
[1] User attempts to access the product prediction model
[2] Face image captured  --> Face Recognition Model
    ├─ Recognised (conf >= 0.30) --> proceed to [3]
    └─ Not recognised            --> ACCESS DENIED
[3] Voice sample captured  --> Voiceprint Verification Model
    ├─ Verified (conf >= 0.30)   --> proceed to [4]
    └─ Not verified              --> ACCESS DENIED
[4] Multimodal gate: Face AND Voice both passed --> identity verified
[5] Product Recommendation Model called
    --> Recommended product category served to verified user
```

### 3.2 How to Run

```bash
# From the repository root
python system_app.py
```

The app auto-discovers enrolled members (those with both face images and
voice samples), loads the three models from `models/`, and runs two demos:
1. **Full authorised transaction** — face + voice → access granted +
   product recommendation
2. **Unauthorized attempt** — demonstrates both denial branches
   (impostor face, impostor voice)

### 3.3 Example Output

```
============================================================
  USER IDENTITY & PRODUCT RECOMMENDATION SYSTEM (CLI)
============================================================
Models loaded: ['face', 'voice', 'product']

Demo customer id: 151

--- DEMO 1: FULL TRANSACTION (authorised user) ---

[1] User attempts to access the product prediction model.
[2] Capturing face image: neutral.jpeg
    Face recognised as: mukasa (confidence 0.91)
[3] Capturing voice sample: confirm_transaction.wav
    Voiceprint matched to: mukasa (confidence 0.99)

[4] Multimodal gate: Face OK + Voice OK -> identity verified.
[5] Calling product prediction model for customer 151 ...
    Recommended product category: Sports (confidence 0.25)

[RESULT] ACCESS GRANTED - authenticated & recommendation served.

--- DEMO 2: UNAUTHORIZED ATTEMPT ---

[Face check]  neutral.jpeg -> nina       conf=0.33  FAIL
[RESULT] ACCESS DENIED - Face not recognised (unauthorized attempt).

[Voice check] confirm_transaction.wav -> samuel conf=0.87  FAIL
[RESULT] ACCESS DENIED - Voice not verified (unauthorized attempt).
```

### 3.4 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Confidence threshold = 0.30 | Balances security (rejecting impostors) with usability (accepting enrolled users); biometric models rarely output near-1.0 confidence on single samples |
| Short-circuit denial | If face recognition fails, the voice check is skipped — access is denied immediately, mirroring a real security checkpoint |
| No recommendation until verified | The product model is invoked **only** after both biometric checks pass, so no purchase suggestion is ever served to an unverified user |
| Notebook + CLI parity | `system_app.py` is also invoked from the Jupyter notebook so the same flow can be demonstrated in either environment |

---

## 4. GitHub Repository

### 4.1 Repository Structure

```
Formative-2---Data-Preprocessing/
├── README.md                              # project overview + how to run
├── CONTRIBUTIONS.md                       # team contribution sheet
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Python/cache/macOS/Jupyter ignores
├── customer_social_profiles - customer_social_profiles.csv
├── customer_transactions - customer_transactions.csv
├── data/
│   ├── customer_social_profiles.xlsx
│   ├── customer_transactions.xlsx
│   ├── merged_dataset.csv                 # cleaned + merged + engineered
│   └── audio/
│       ├── raw/<member>/<phrase>.wav|.ogg
│       ├── augmented/<member>/
│       └── audio_features.csv
├── image_processing/
│   ├── images/<member>/<expression>.jpeg
│   ├── augmented_images/<member>/
│   ├── output/image_features.csv
│   └── image_preprocessing.ipynb
├── models/
│   ├── product_model.joblib               # product recommendation (product team)
│   ├── product_metrics.json               # product model metrics
│   ├── face_model.joblib                  # face recognition (Member 4)
│   └── voice_model.joblib                 # voiceprint verification (Member 4)
├── plots/                                 # EDA + model evaluation figures
├── system_app.py                          # CLI app (Member 4)
├── train_member4_models.py                # face + voice model training (Member 4)
├── REPORT_MEMBER4.md                      # Member 4 report section
└── VIDEO_STORYBOARD.md                    # demo video plan (Member 4)
```

### 4.2 Files to Commit (Member 4's additions)

```bash
git add system_app.py
git add train_member4_models.py
git add models/face_model.joblib
git add models/voice_model.joblib
git add REPORT_MEMBER4.md
git add VIDEO_STORYBOARD.md
git commit -m "Member 4: CLI app, face+voice models, report, video storyboard"
git push origin main
```

**Note:** Do not commit large raw media files (`.wav`, `.ogg`, `.jpeg`) unless necessary — commit the processed CSVs (`audio_features.csv`, `image_features.csv`, `merged_dataset.csv`) and models instead. List raw media in `.gitignore` if files are large.

---

## 5. Demo Video — Storyboard

### 5.1 What the Video Must Show

- Launching the CLI app (`python system_app.py`)
- All three models loading (face, voice, product)
- **Demo 1 — Full authorised transaction:** face image + voice sample → access
  granted → product recommendation displayed
- **Demo 2 — Unauthorized attempt:** impostor face and/or impostor voice →
  access denied (both denial branches shown)
- The multimodal gate rule in action: one failed check = denial, no
  recommendation served

### 5.2 Storyboard (Suggested Sequence)

| # | Scene | Action / Narration |
|---|-------|-------------------|
| 1 | Intro | Show the repository structure; open terminal; run `python system_app.py` |
| 2 | Models loaded | Terminal prints the three loaded models (face, voice, product) |
| 3 | Demo 1 start | Present an enrolled member's face image and voice sample |
| 4 | Demo 1 pass | Face recognised + voice verified → multimodal gate passes → product category recommended |
| 5 | Demo 2 start | Present an impostor face/voice pair |
| 6 | Demo 2 deny | Face or voice fails → `ACCESS DENIED`, no recommendation served |
| 7 | Close | Summarise: verified users only receive product recommendations |

### 5.3 Recording Instructions

- Use a screen recorder (OBS, Loom, QuickTime, or built-in Windows recorder)
- Record in a quiet environment; speak clearly and narrate each step
- Keep the terminal font large and legible; zoom in if needed
- Target length: 2–4 minutes
- Upload the video (YouTube unlisted / Google Drive / Stream) and paste the link
  into the team report

---

## 6. Integration Checklist

- [ ] Face recognition model trained and saved (`models/face_model.joblib`)
- [ ] Voiceprint verification model trained and saved (`models/voice_model.joblib`)
- [ ] Product recommendation model available (`models/product_model.joblib`)
- [ ] CLI app runs end-to-end (`python system_app.py`)
- [ ] Authorised transaction grants access + serves recommendation
- [ ] Unauthorised face attempt → access denied
- [ ] Unauthorised voice attempt → access denied
- [ ] Report section written (`REPORT_MEMBER4.md`)
- [ ] Video storyboard written (`VIDEO_STORYBOARD.md`)
- [ ] All Member 4 files committed and pushed to GitHub
