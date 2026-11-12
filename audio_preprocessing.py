"""Augment voice samples, visualize them, and extract audio features."""

import argparse
import hashlib
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf


def _stable_seed(path: Path) -> int:
    return int.from_bytes(hashlib.sha256(str(path).encode("utf-8")).digest()[:4], "little")


def _extract_features(
    audio: np.ndarray,
    sample_rate: int,
    member: str,
    phrase: str,
    variant: str,
    source: str,
) -> dict[str, object]:
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
    rms = librosa.feature.rms(y=audio)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)

    features: dict[str, object] = {
        "member": member,
        "phrase": phrase,
        "variant": variant,
        "source": source,
        "sample_rate": sample_rate,
        "duration_seconds": len(audio) / sample_rate,
        "spectral_rolloff_mean_hz": float(np.mean(rolloff)),
        "rms_energy_mean": float(np.mean(rms)),
        "spectral_centroid_mean_hz": float(np.mean(centroid)),
        "zero_crossing_rate_mean": float(np.mean(zero_crossing_rate)),
    }
    for index, coefficient in enumerate(mfcc, start=1):
        features[f"mfcc_{index:02d}_mean"] = float(np.mean(coefficient))
        features[f"mfcc_{index:02d}_std"] = float(np.std(coefficient))
    return features


def _make_augmentations(audio: np.ndarray, sample_rate: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=audio.shape)
    signal_rms = float(np.sqrt(np.mean(np.square(audio))))
    noise_rms = float(np.sqrt(np.mean(np.square(noise))))
    noise = noise * (signal_rms / (noise_rms * 10.0))

    return {
        "pitch_up": librosa.effects.pitch_shift(y=audio, sr=sample_rate, n_steps=2),
        "time_stretch": librosa.effects.time_stretch(y=audio, rate=0.9),
        "background_noise": audio + noise,
    }


def _save_visualization(audio: np.ndarray, sample_rate: int, path: Path, title: str) -> None:
    times = np.arange(len(audio)) / sample_rate
    spectrum = np.abs(librosa.stft(audio))
    spectrum_db = librosa.amplitude_to_db(spectrum, ref=np.max)
    time_axis = librosa.frames_to_time(
        np.arange(spectrum_db.shape[1]), sr=sample_rate
    )
    frequency_axis = librosa.fft_frequencies(sr=sample_rate)

    figure, (waveform_axis, spectrum_axis) = plt.subplots(
        2, 1, figsize=(11, 7), constrained_layout=True
    )
    waveform_axis.plot(times, audio, linewidth=0.6)
    waveform_axis.set(title=f"{title} — waveform", xlabel="Time (s)", ylabel="Amplitude")
    image = spectrum_axis.imshow(
        spectrum_db,
        origin="lower",
        aspect="auto",
        extent=[
            time_axis[0],
            time_axis[-1] if len(time_axis) > 1 else len(audio) / sample_rate,
            frequency_axis[0],
            frequency_axis[-1],
        ],
        cmap="magma",
    )
    spectrum_axis.set(
        title=f"{title} — spectrogram",
        xlabel="Time (s)",
        ylabel="Frequency (Hz)",
    )
    figure.colorbar(image, ax=spectrum_axis, label="Amplitude (dB)")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def process_audio(input_dir: Path, output_dir: Path) -> pd.DataFrame:
    recordings = sorted(input_dir.rglob("*.wav"))
    if not recordings:
        raise FileNotFoundError(f"No .wav recordings found under {input_dir}")

    rows: list[dict[str, object]] = []
    for recording in recordings:
        relative_path = recording.relative_to(input_dir)
        member = relative_path.parent.name
        phrase = recording.stem
        audio, sample_rate = librosa.load(recording, sr=None, mono=True)
        if audio.size == 0:
            raise ValueError(f"Recording is empty: {recording}")

        rows.append(
            _extract_features(
                audio, sample_rate, member, phrase, "original", relative_path.as_posix()
            )
        )
        plot_path = output_dir / "plots" / member / f"{recording.stem}.png"
        _save_visualization(
            audio, sample_rate, plot_path, f"{member} — {phrase} (original)"
        )

        augmentations = _make_augmentations(audio, sample_rate, _stable_seed(recording))
        for variant, augmented_audio in augmentations.items():
            augmented_path = (
                output_dir
                / "augmented"
                / member
                / f"{recording.stem}__{variant}.wav"
            )
            augmented_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(augmented_path, augmented_audio, sample_rate)
            rows.append(
                _extract_features(
                    augmented_audio,
                    sample_rate,
                    member,
                    phrase,
                    variant,
                    augmented_path.relative_to(output_dir).as_posix(),
                )
            )

    features = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "audio_features.csv"
    features.to_csv(csv_path, index=False)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create audio augmentations, plots, and an audio feature CSV."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/audio/raw"),
        help="Directory containing member subfolders with WAV recordings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/audio"),
        help="Directory for augmentations, plots, and audio_features.csv.",
    )
    args = parser.parse_args()

    features = process_audio(args.input_dir, args.output_dir)
    print(f"Processed {len(features)} original and augmented audio samples.")
    print(f"Features saved to {args.output_dir / 'audio_features.csv'}")
    print(f"Waveform and spectrogram plots saved to {args.output_dir / 'plots'}")


if __name__ == "__main__":
    main()
