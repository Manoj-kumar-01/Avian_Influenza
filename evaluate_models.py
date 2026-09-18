import os
import glob
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from audio_utils import extract_features, extract_gatekeeper_features
from cnn_model import AudioCNN
from hen_filter_model import HenVoiceFilter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(BASE_DIR, "Inputs")
DATASET_PATH = os.path.join(INPUTS_DIR, "Chicken_Audio_Dataset")
NON_HEN_DIR = os.path.join(INPUTS_DIR, "Non_Hen_Dataset")

CNN_MODEL_PATH = os.path.join(BASE_DIR, "cnn_model.pth")
OOD_MODEL_PATH = os.path.join(BASE_DIR, "ood_detector.pkl")

CLASSES = ["Healthy", "Unhealthy", "Noise"]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}

def run_evaluation():
    print("=" * 65)
    print("  COMPREHENSIVE MODEL EVALUATION & VALIDATION SUITE")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Evaluate Stage 1: Hen Voice Filter
    print("\n[STAGE 1 EVALUATION] Hen Voice Filter / OOD Gatekeeper")
    print("-" * 55)
    if not os.path.exists(OOD_MODEL_PATH):
        print(f"[SKIP] Gatekeeper model not found at {OOD_MODEL_PATH}")
    else:
        gatekeeper = HenVoiceFilter.load(OOD_MODEL_PATH)
        hen_files = glob.glob(os.path.join(DATASET_PATH, "Healthy", "*.wav"))[:25] + \
                    glob.glob(os.path.join(DATASET_PATH, "Unhealthy", "*.wav"))[:25]
        non_hen_files = glob.glob(os.path.join(NON_HEN_DIR, "*.wav")) + \
                        glob.glob(os.path.join(DATASET_PATH, "Noise", "*.wav"))[:25]

        correct = 0
        total = 0
        for f in hen_files:
            feat = extract_gatekeeper_features(f)
            if feat is not None:
                res = gatekeeper.predict(feat)
                if res["is_hen"]:
                    correct += 1
                total += 1

        for f in non_hen_files:
            feat = extract_gatekeeper_features(f)
            if feat is not None:
                res = gatekeeper.predict(feat)
                if not res["is_hen"]:
                    correct += 1
                total += 1

        gate_acc = (correct / total) * 100.0 if total > 0 else 0.0
        print(f"Stage 1 Gatekeeper Discrimination Accuracy: {gate_acc:.2f}% ({correct}/{total})")

    # 2. Evaluate Stage 2: CNN Avian Influenza Classifier
    print("\n[STAGE 2 EVALUATION] Deep ResNet Avian Influenza Classifier")
    print("-" * 55)
    if not os.path.exists(CNN_MODEL_PATH):
        print(f"[ERROR] CNN model not found at {CNN_MODEL_PATH}")
        return

    model = AudioCNN(num_classes=3, backbone="resnet34").to(device)
    model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    for cls in CLASSES:
        folder = os.path.join(DATASET_PATH, cls)
        files = glob.glob(os.path.join(folder, "*.wav"))
        for f in files:
            spec = extract_features(f)
            if spec is not None:
                spec_tensor = spec.unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(spec_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)[0]
                    pred = torch.argmax(probs).item()
                    
                all_preds.append(pred)
                all_targets.append(CLASS_TO_IDX[cls])
                all_probs.append(probs.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred) * 100.0
    print(f"\nOverall Dataset Classification Accuracy: {acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(f"{'':12s} {'Pred Healthy':14s} {'Pred Unhealthy':14s} {'Pred Noise':14s}")
    for idx, row in enumerate(cm):
        print(f"Actual {CLASSES[idx]:9s}: {row[0]:<14d} {row[1]:<14d} {row[2]:<14d}")

    # Avian Influenza detection specific metrics
    # Class index 1 is Unhealthy (Avian Influenza)
    unhealthy_idx = CLASS_TO_IDX["Unhealthy"]
    ai_tp = cm[unhealthy_idx, unhealthy_idx]
    ai_fn = np.sum(cm[unhealthy_idx, :]) - ai_tp
    ai_fp = np.sum(cm[:, unhealthy_idx]) - ai_tp
    ai_recall = ai_tp / (ai_tp + ai_fn) if (ai_tp + ai_fn) > 0 else 0
    ai_precision = ai_tp / (ai_tp + ai_fp) if (ai_tp + ai_fp) > 0 else 0

    print(f"\nAvian Influenza Sensitivity / Recall: {ai_recall * 100:.2f}%")
    print(f"Avian Influenza Specificity / Precision: {ai_precision * 100:.2f}%")

    if acc >= 95.0:
        print("\n[SUCCESS] Goal Achieved: Accuracy exceeds 95% requirement!")
    else:
        print(f"\n[INFO] Current Accuracy is {acc:.2f}%.")

if __name__ == "__main__":
    run_evaluation()
