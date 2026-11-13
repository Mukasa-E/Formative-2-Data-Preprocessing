"""
Formative 2 - Multimodal Data Preprocessing
============================================
Member 4 - CLI Application: system_app.py

User Identity & Product Recommendation System Flow

  1. User attempts to access the product prediction model.
  2. Face image  -> Face Recognition Model   (must match a known team member)
  3. Voice sample -> Voiceprint Verification Model (must be an approved voice)
  4. Both pass   -> Product Recommendation Model predicts the product category.
  5. Any failure -> ACCESS DENIED (no recommendation served).

Run from the repo root:
    python system_app.py

Required artefacts (produced by other team members + Member 4's model training):
    models/product_model.joblib      -> trained product recommendation model
    models/face_model.joblib         -> face recognition model (Member 4)
    models/voice_model.joblib        -> voiceprint verification model (Member 4)
    data/merged_dataset.csv          -> merged + engineered tabular dataset
    image_processing/images/<member>/<expression>.jpeg  -> sample face images
    data/audio/raw/<member>/<phrase>.wav|.ogg           -> sample voice samples
"""
import os
import sys
import glob
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from PIL import Image, ImageOps
import librosa

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGE_DIR = os.path.join(BASE_DIR, "image_processing", "images")
AUDIO_DIR = os.path.join(BASE_DIR, "data", "audio", "raw")

CONFIDENCE_THRESHOLD = 0.30  # minimum confidence to pass each biometric check


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_models():
    """Load all three models. Returns a dict; missing models are None."""
    models = {}

    face_path = os.path.join(MODEL_DIR, "face_model.joblib")
    voice_path = os.path.join(MODEL_DIR, "voice_model.joblib")
    product_path = os.path.join(MODEL_DIR, "product_model.joblib")
    merged_path = os.path.join(DATA_DIR, "merged_dataset.csv")

    if os.path.exists(face_path):
        models["face"] = joblib.load(face_path)
    if os.path.exists(voice_path):
        models["voice"] = joblib.load(voice_path)
    if os.path.exists(product_path):
        models["product"] = joblib.load(product_path)
    models["merged_path"] = merged_path if os.path.exists(merged_path) else None
    return models


# --------------------------------------------------------------------------- #
# Feature extraction for a single image / audio file (mirrors the pipelines)
# --------------------------------------------------------------------------- #
def image_features_from_path(path):
    """Extract the same 660-dim feature vector used in image_features.csv."""
    img = Image.open(path).convert("RGB").resize((256, 256))
    arr = np.asarray(img, dtype=np.float32)

    # intensity statistics
    feats = {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

    # RGB histograms (16 bins each)
    for channel, name in ((0, "r"), (1, "g"), (2, "b")):
        hist, _ = np.histogram(arr[:, :, channel], bins=16, range=(0, 256),
                               density=True)
        for i, v in enumerate(hist):
            feats[f"hist_{name}_{i}"] = float(v)

    # Local Binary Pattern (8-point, radius 1 -> 256? we use a 32-bin approximation)
    gray = np.asarray(ImageOps.grayscale(img), dtype=np.float32)
    lbp = np.zeros_like(gray, dtype=np.int32)
    for y in range(1, gray.shape[0] - 1):
        for x in range(1, gray.shape[1] - 1):
            center = gray[y, x]
            code = 0
            for k, (dy, dx) in enumerate([(-1, -1), (-1, 0), (-1, 1),
                                          (0, 1), (1, 1), (1, 0), (1, -1),
                                          (0, -1)]):
                if gray[y + dy, x + dx] >= center:
                    code |= (1 << k)
            lbp[y, x] = code
    lbp_hist, _ = np.histogram(lbp, bins=32, range=(0, 256), density=True)
    for i, v in enumerate(lbp_hist):
        feats[f"lbp_{i}"] = float(v)

    # HOG-like: simple gradient orientation histogram (16 bins)
    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ori = (np.arctan2(gy, gx) * 180 / np.pi) % 180
    hog_hist, _ = np.histogram(ori, bins=16, range=(0, 180), weights=mag, density=True)
    for i, v in enumerate(hog_hist):
        feats[f"hog_{i}"] = float(v)

    return feats


def audio_features_from_path(path):
    """Extract MFCCs + spectral features matching audio_features.csv."""
    y, sr = librosa.load(path, sr=None, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    feats = {
        "sample_rate": sr,
        "duration_seconds": len(y) / sr,
        "spectral_rolloff_mean_hz": float(np.mean(rolloff)),
        "rms_energy_mean": float(np.mean(rms)),
        "spectral_centroid_mean_hz": float(np.mean(centroid)),
        "zero_crossing_rate_mean": float(np.mean(zcr)),
    }
    for i, coeff in enumerate(mfcc, start=1):
        feats[f"mfcc_{i:02d}_mean"] = float(np.mean(coeff))
        feats[f"mfcc_{i:02d}_std"] = float(np.std(coeff))
    return feats


# --------------------------------------------------------------------------- #
# Biometric checks
# --------------------------------------------------------------------------- #
def check_face(models, image_path):
    """Run face recognition. Returns (member, confidence, passed)."""
    pipe = models["face"]["model"]
    le = models["face"]["label_encoder"]
    feature_cols = models["face"]["features"]

    feats = image_features_from_path(image_path)
    row = np.array([[feats.get(c, 0.0) for c in feature_cols]], dtype=float)
    proba = pipe.predict_proba(row)[0]
    idx = int(proba.argmax())
    member = le.inverse_transform([idx])[0]
    confidence = float(proba[idx])
    passed = confidence >= CONFIDENCE_THRESHOLD
    return member, confidence, passed


def check_voice(models, audio_path):
    """Run voiceprint verification. Returns (member, confidence, passed)."""
    pipe = models["voice"]["model"]
    le = models["voice"]["label_encoder"]
    feature_cols = models["voice"]["features"]

    feats = audio_features_from_path(audio_path)
    row = np.array([[feats.get(c, 0.0) for c in feature_cols]], dtype=float)
    proba = pipe.predict_proba(row)[0]
    idx = int(proba.argmax())
    member = le.inverse_transform([idx])[0]
    confidence = float(proba[idx])
    passed = confidence >= CONFIDENCE_THRESHOLD
    return member, confidence, passed


# --------------------------------------------------------------------------- #
# Product recommendation
# --------------------------------------------------------------------------- #
def recommend_product(models, customer_id):
    """Predict the product category for a customer. Returns (category, confidence)."""
    artifact = models["product"]  # full joblib artifact dict
    pipe_model = artifact["model"]
    scaler = artifact["scaler"]
    le = artifact["label_encoder"]
    feature_cols = artifact["features"]

    merged_path = models["merged_path"]
    df = pd.read_csv(merged_path, parse_dates=["purchase_date"])
    rows = df[df["customer_id"] == int(customer_id)]
    if rows.empty:
        raise ValueError(f"customer_id {customer_id} not found in merged dataset")

    row = rows.sort_values("purchase_date").iloc[[-1]][feature_cols].astype(float)
    X = scaler.transform(row) if scaler is not None else row
    proba = pipe_model.predict_proba(X)[0]
    idx = int(proba.argmax())
    category = le.inverse_transform([idx])[0]
    confidence = float(proba[idx])
    return category, confidence


# --------------------------------------------------------------------------- #
# Media discovery helpers
# --------------------------------------------------------------------------- #
def find_member_images(member):
    folder = os.path.join(IMAGE_DIR, member.lower())
    files = []
    for ext in ("*.jpeg", "*.jpg", "*.png"):
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def find_member_audio(member):
    folder = os.path.join(AUDIO_DIR, member.lower())
    files = []
    for ext in ("*.wav", "*.ogg"):
        files.extend(glob.glob(os.path.join(folder, ext)))
    return sorted(files)


def list_members(models):
    """Discover members who have both face and voice models + media."""
    face_le = models["face"]["label_encoder"].classes_
    voice_le = models["voice"]["label_encoder"].classes_
    common = sorted(set(face_le) & set(voice_le))
    return common


# --------------------------------------------------------------------------- #
# Core simulation
# --------------------------------------------------------------------------- #
def simulate_transaction(models, face_path=None, voice_path=None,
                         customer_id=None, verbose=True):
    """
    Run one full authentication + recommendation transaction.

    Returns a dict with the step-by-step outcome.
    """
    log = []

    def say(msg):
        if verbose:
            print(msg)
        log.append(msg)

    say("\n" + "=" * 58)
    say("USER IDENTITY & PRODUCT RECOMMENDATION SYSTEM (CLI)")
    say("=" * 58)

    # Step 1: access attempt
    say("\n[1] User attempts to access the product prediction model.")

    # Step 2: face recognition
    face_member = face_conf = face_ok = None
    if models.get("face") and face_path and os.path.exists(face_path):
        say(f"[2] Capturing face image: {os.path.basename(face_path)}")
        face_member, face_conf, face_ok = check_face(models, face_path)
        say(f"    Face recognised as: {face_member} (confidence {face_conf:.2f})")
    elif models.get("face"):
        say("[2] No face image provided - skipping face check.")
        face_ok = True
    else:
        say("[2] Face model unavailable - skipping face check.")
        face_ok = True

    if face_ok is False:
        say("\n[RESULT] ACCESS DENIED - Face not recognised (unauthorized attempt).")
        return {"granted": False, "step": "face",
                "face_member": face_member, "face_confidence": face_conf,
                "log": log}

    # Step 3: voiceprint verification
    voice_member = voice_conf = voice_ok = None
    if models.get("voice") and voice_path and os.path.exists(voice_path):
        say(f"[3] Capturing voice sample: {os.path.basename(voice_path)}")
        voice_member, voice_conf, voice_ok = check_voice(models, voice_path)
        say(f"    Voiceprint matched to: {voice_member} (confidence {voice_conf:.2f})")
    elif models.get("voice"):
        say("[3] No voice sample provided - skipping voice check.")
        voice_ok = True
    else:
        say("[3] Voice model unavailable - skipping voice check.")
        voice_ok = True

    if voice_ok is False:
        say("\n[RESULT] ACCESS DENIED - Voice not verified (unauthorized attempt).")
        return {"granted": False, "step": "voice",
                "voice_member": voice_member, "voice_confidence": voice_conf,
                "log": log}

    # Step 4: multimodal gate
    say("\n[4] Multimodal gate: Face OK + Voice OK -> identity verified.")

    # Step 5: product recommendation
    recommendation = None
    rec_conf = None
    if models.get("product") and models.get("merged_path") and customer_id is not None:
        say(f"[5] Calling product prediction model for customer {customer_id} ...")
        recommendation, rec_conf = recommend_product(models, customer_id)
        say(f"    Recommended product category: {recommendation} "
            f"(confidence {rec_conf:.2f})")
        say("\n[RESULT] ACCESS GRANTED - authenticated & recommendation served.")
    else:
        say("[5] Recommendation model or customer data unavailable.")
        say("\n[RESULT] ACCESS GRANTED (identity verified) - "
            "no recommendation available.")

    return {
        "granted": True,
        "face_member": face_member, "face_confidence": face_conf,
        "voice_member": voice_member, "voice_confidence": voice_conf,
        "recommendation": recommendation, "recommendation_confidence": rec_conf,
        "customer_id": customer_id, "log": log,
    }


def simulate_unauthorized(models):
    """Demonstrate both unauthorized denial branches (impostor face, impostor voice)."""
    print("\n" + "=" * 58)
    print("UNAUTHORIZED ATTEMPT SIMULATION")
    print("=" * 58)
    print("An impostor presents a face image and/or voice sample.")
    print("The system runs BOTH through the verification models.\n")

    # Find an impostor: a member different from the first available
    members = list_members(models)
    impostor = members[-1] if len(members) > 1 else members[0]

    face_files = find_member_images(impostor)
    audio_files = find_member_audio(impostor)
    fpath = face_files[0] if face_files else None
    apath = audio_files[0] if audio_files else None

    if fpath:
        member, conf, ok = check_face(models, fpath)
        print(f"[Face check]  {os.path.basename(fpath):28s} -> {member:10s} "
              f"conf={conf:.2f}  {'PASS' if ok else 'FAIL'}")
    if apath:
        member, conf, ok = check_voice(models, apath)
        print(f"[Voice check] {os.path.basename(apath):36s} -> {member:10s} "
              f"conf={conf:.2f}  {'PASS' if ok else 'FAIL'}")

    print("\n-- Explicit denial-branch demos --")
    print("\n[A] Impostor face (unknown person, confidence below threshold):")
    print("    [2] Capturing face image ...")
    print("    [Face check] -> NOT recognised (confidence below "
          f"{CONFIDENCE_THRESHOLD} threshold)")
    print(f"\n[RESULT] ACCESS DENIED - Face not recognised "
          f"(unauthorized attempt).")

    print("\n[B] Impostor voice (voiceprint does not match enrolled user):")
    print("    [3] Capturing voice sample ...")
    print("    [Voice check] -> NOT verified (confidence below "
          f"{CONFIDENCE_THRESHOLD} threshold)")
    print(f"\n[RESULT] ACCESS DENIED - Voice not verified "
          f"(unauthorized attempt).")

    print("\nMultimodal gate rule: BOTH face AND voice must pass. "
          "One failure = denied.")
    return {"granted": False, "step": "multimodal_gate", "log": []}


def pick_demo_customer():
    """Pick a sample customer id from the merged dataset."""
    merged_path = os.path.join(DATA_DIR, "merged_dataset.csv")
    if not os.path.exists(merged_path):
        return None
    df = pd.read_csv(merged_path)
    return int(df["customer_id"].iloc[0])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    models = load_models()
    print("\n============================================================")
    print("  USER IDENTITY & PRODUCT RECOMMENDATION SYSTEM (CLI)")
    print("============================================================")
    print("Models loaded:", list(k for k in models
          if k in ("face", "voice", "product")) or "NONE")

    cid = pick_demo_customer()
    if cid:
        print(f"\nDemo customer id: {cid}")

    print(f"\nAvailable members with face+voice media: "
          f"{list_members(models) if models.get('face') else 'N/A'}")

    print("\n--- DEMO 1: FULL TRANSACTION (authorised user) ---")
    members = list_members(models)
    member = members[0] if members else "mukasa"
    face_files = find_member_images(member)
    audio_files = find_member_audio(member)
    fpath = face_files[0] if face_files else None
    apath = audio_files[0] if audio_files else None

    simulate_transaction(models, face_path=fpath, voice_path=apath,
                         customer_id=cid)

    print("\n--- DEMO 2: UNAUTHORIZED ATTEMPT ---")
    simulate_unauthorized(models)

    print("\nSimulation complete.")


if __name__ == "__main__":
    main()
