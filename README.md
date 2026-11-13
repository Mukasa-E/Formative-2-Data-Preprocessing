# Formative 2 - Data Preprocessing

## Audio preprocessing

Place the two WAV or OGG recordings for each team member in a member-specific folder.
Use descriptive filenames such as `yes_approve.wav` and
`confirm_transaction.wav` (OGG files are also supported):

```text
data/audio/raw/
└── mukasa/
    ├── yes_approve.wav
    └── confirm_transaction.wav
```

Run the audio pipeline from the repository root:

```powershell
python audio_preprocessing.py
```

The script processes every `.wav` and `.ogg` file recursively. For each recording, it
creates pitch-shifted, time-stretched, and background-noise augmented versions,
plots the original waveform and spectrogram, and extracts MFCC means and
standard deviations plus spectral roll-off, RMS energy, spectral centroid, and
zero-crossing rate. Outputs are saved under `data/audio/`:

- `audio_features.csv` contains one row for each original or augmented sample.
- `augmented/<member>/` contains the augmented WAV files.
- `plots/<member>/` contains a waveform and spectrogram figure for each original.

Use the member's agreed identifier as the folder name. The feature table also
retains the phrase and augmentation variant for traceability.

## Image preprocessing

Place three facial photos for each team member in a lowercase, member-specific
folder (the same ID used for audio). Accepted formats are `.jpg`, `.jpeg` and
`.png`:

```text
image_processing/images/
└── mukasa/
    ├── neutral.jpeg
    ├── smiling.jpeg
    └── surprised.jpeg
```

Install the dependencies and run the notebook from the `image_processing/`
folder:

```bash
pip install -r requirements.txt
cd image_processing
jupyter nbconvert --to notebook --execute --inplace image_preprocessing.ipynb
```

You can also open `image_preprocessing.ipynb` in Jupyter or VS Code and choose
Run All. The notebook finds members automatically and flags any missing
expression. It pads and resizes every image to 256×256, then applies four
augmentations: horizontal flip, 15° rotation, grayscale and
brightness/contrast. Next it detects and crops each face to 128×128 using
OpenCV Haar cascades. Finally it extracts 660 features per image: intensity
statistics, RGB colour histograms, Local Binary Pattern texture and HOG
shape. Outputs are saved under `image_processing/`:

- `augmented_images/<member>/<expression>_<augmentation>.jpg` contains the
  augmented images.
- `output/image_features.csv` contains one row for each original or augmented
  image, with `member_id`, `expression`, `augmentation`, `face_detected` and
  the feature columns.

OpenCV must be version 4.x (`opencv-python<5`, pinned in `requirements.txt`),
because OpenCV 5 no longer includes the Haar cascade face detector.

---

## System Simulation (CLI App) — Member 4

The command-line application `system_app.py` demonstrates the full
User Identity & Product Recommendation System flow:

1. **Face image** → Face Recognition Model (recognise a known team member)
2. **Voice sample** → Voiceprint Verification Model (approve an enrolled voice)
3. **Both pass** → Product Recommendation Model predicts the product category
4. **Any failure** → ACCESS DENIED (no recommendation served)

### Run the CLI app

```bash
python system_app.py
```

The app loads all three models from `models/` and runs two demos:
- **Demo 1 — Full authorised transaction:** face + voice → access granted +
  product recommendation served
- **Demo 2 — Unauthorized attempt:** impostor face/voice → access denied
  (both denial branches shown)

### Models

| Model | File | Trained by |
|-------|------|------------|
| Face Recognition | `models/face_model.joblib` | Member 4 |
| Voiceprint Verification | `models/voice_model.joblib` | Member 4 |
| Product Recommendation | `models/product_model.joblib` | Product team |

### Deliverables

- `system_app.py` — CLI system simulation
- `train_member4_models.py` — trains face + voice models from existing features
- `REPORT_MEMBER4.md` — Member 4's report section
- `VIDEO_STORYBOARD.md` — demo video production plan & storyboard

