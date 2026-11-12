# Formative 2 - Data Preprocessing

## Audio preprocessing

Place the two WAV recordings for each team member in a member-specific folder.
Use descriptive filenames such as `yes_approve.wav` and
`confirm_transaction.wav`:

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

The script processes every `.wav` file recursively. For each recording, it
creates pitch-shifted, time-stretched, and background-noise augmented versions,
plots the original waveform and spectrogram, and extracts MFCC means and
standard deviations plus spectral roll-off, RMS energy, spectral centroid, and
zero-crossing rate. Outputs are saved under `data/audio/`:

- `audio_features.csv` contains one row for each original or augmented sample.
- `augmented/<member>/` contains the augmented WAV files.
- `plots/<member>/` contains a waveform and spectrogram figure for each original.

Use the member's agreed identifier as the folder name. The feature table also
retains the phrase and augmentation variant for traceability.
