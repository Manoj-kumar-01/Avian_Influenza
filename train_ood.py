import os
import glob
import numpy as np
from audio_utils import extract_gatekeeper_features
from hen_filter_model import HenVoiceFilter
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(BASE_DIR, "Inputs")
CHICKEN_DIR = os.path.join(INPUTS_DIR, "Chicken_Audio_Dataset")
NON_HEN_DIR = os.path.join(INPUTS_DIR, "Non_Hen_Dataset")
OOD_SAVE_PATH = os.path.join(BASE_DIR, "ood_detector.pkl")

def train_gatekeeper():
    print("=" * 65)
    print("  Training Stage 1 Hen Voice Filter & OOD Gatekeeper")
    print("=" * 65)

    # 1. Collect Hen Vocalizations (Healthy + Unhealthy)
    hen_files = []
    for cat in ["Healthy", "Unhealthy"]:
        cat_dir = os.path.join(CHICKEN_DIR, cat)
        files = glob.glob(os.path.join(cat_dir, "*.wav"))
        hen_files.extend(files)
        print(f"Loaded {len(files)} files from {cat}")

    # 2. Collect Non-Hen Audios (Noise + Non_Hen_Dataset)
    non_hen_files = []
    noise_dir = os.path.join(CHICKEN_DIR, "Noise")
    if os.path.exists(noise_dir):
        files = glob.glob(os.path.join(noise_dir, "*.wav"))
        non_hen_files.extend(files)
        print(f"Loaded {len(files)} files from Environmental Noise")

    if os.path.exists(NON_HEN_DIR):
        files = glob.glob(os.path.join(NON_HEN_DIR, "*.wav"))
        non_hen_files.extend(files)
        print(f"Loaded {len(files)} files from Non-Hen Audio Dataset")

    print(f"\nTotal Hen files: {len(hen_files)}, Total Non-Hen files: {len(non_hen_files)}")

    if len(hen_files) == 0:
        print("[ERROR] No hen files found. Please run download_datasets.py first.")
        return

    # Extract features
    print("\nExtracting acoustic features for Hen Vocalizations...")
    hen_features = []
    for f in hen_files:
        feat = extract_gatekeeper_features(f)
        if feat is not None:
            hen_features.append(feat)
    X_hen = np.array(hen_features)
    print(f"Extracted {X_hen.shape[0]} hen feature vectors (dim: {X_hen.shape[1]}).")

    print("\nExtracting acoustic features for Non-Hen Audio...")
    non_hen_features = []
    for f in non_hen_files:
        feat = extract_gatekeeper_features(f)
        if feat is not None:
            non_hen_features.append(feat)
    X_non_hen = np.array(non_hen_features)
    print(f"Extracted {X_non_hen.shape[0]} non-hen feature vectors.")

    # Train HenVoiceFilter
    gatekeeper = HenVoiceFilter(n_estimators=200)
    gatekeeper.fit(X_hen, X_non_hen)

    # Validate
    y_true = np.concatenate([np.ones(len(X_hen)), np.zeros(len(X_non_hen))])
    X_all = np.vstack([X_hen, X_non_hen])
    
    preds = []
    for i in range(len(X_all)):
        res = gatekeeper.predict(X_all[i])
        preds.append(1 if res["is_hen"] else 0)
    y_pred = np.array(preds)

    acc = accuracy_score(y_true, y_pred)
    print(f"\nStage 1 Gatekeeper Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Non-Hen Audio", "Hen Vocalization"]))

    # Save artifact
    gatekeeper.save(OOD_SAVE_PATH)
    print(f"\nSaved gatekeeper model to: {OOD_SAVE_PATH}")

if __name__ == "__main__":
    train_gatekeeper()
