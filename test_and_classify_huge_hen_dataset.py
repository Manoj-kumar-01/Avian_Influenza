import os
import glob
import pickle
import torch
import numpy as np
from audio_utils import extract_gatekeeper_features, extract_features
from cnn_model import AudioCNN
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(BASE_DIR, "Inputs")
TEST_HEN_DIR = os.path.join(INPUTS_DIR, "Huge_Normal_Hen_Dataset")
CNN_MODEL_PATH = os.path.join(BASE_DIR, "cnn_model.pth")
OOD_SAVE_PATH = os.path.join(BASE_DIR, "ood_detector.pkl")
CLASSES = ["Healthy", "Unhealthy", "Noise"]

def run_large_hen_evaluation():
    print("=" * 70)
    print("  MASSIVE NORMAL HEN BENCHMARK EVALUATION & CLASSIFICATION")
    print("=" * 70)
    
    # 1. Load Models
    print("Loading models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    from hen_filter_model import HenVoiceFilter
    gatekeeper = HenVoiceFilter.load(OOD_SAVE_PATH)
    print("Stage 1 Hen Voice Filter / Gatekeeper loaded.")
    
    cnn_model = AudioCNN(num_classes=len(CLASSES), backbone="resnet34").to(device)
    cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
    cnn_model.eval()
    print("Stage 2 Deep ResNet Disease Classifier loaded.")
    
    # 2. Gather all normal hen audio files
    hen_files = sorted(glob.glob(os.path.join(TEST_HEN_DIR, "*.wav")))
    print(f"\nFound {len(hen_files)} authentic normal hen audio files to test.")
    if len(hen_files) == 0:
        print("No files found yet in Huge_Normal_Hen_Dataset.")
        return

    # Results tracking
    stage1_passed = 0
    stage1_rejected = 0
    
    stage2_predictions = []
    confidences = []
    
    detailed_results = []
    
    print("\nRunning inference on all normal hen samples...")
    for idx, fpath in enumerate(hen_files):
        fname = os.path.basename(fpath)
        
        # Stage 1: Gatekeeper
        gate_feat = extract_gatekeeper_features(fpath)
        if gate_feat is None:
            continue
        gate_res = gatekeeper.predict(gate_feat)
        is_hen = gate_res["is_hen"]
        gate_prob = gate_res["hen_probability"]
        
        if not is_hen:
            stage1_rejected += 1
            detailed_results.append({
                "file": fname,
                "stage1": "Rejected (Non-Hen)",
                "stage1_prob": gate_prob,
                "stage2": "Skipped",
                "confidence": 0.0
            })
            continue
            
        stage1_passed += 1
        
        # Stage 2: Deep ResNet Classifier
        spec = extract_features(fpath)
        if spec is None:
            continue
            
        spec_tensor = spec.unsqueeze(0).to(device) # (1, 1, 128, T)
        with torch.no_grad():
            outputs = cnn_model(spec_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)
            pred_class = CLASSES[pred_idx]
            conf = float(probs[pred_idx])
            
        stage2_predictions.append(pred_class)
        confidences.append(conf)
        
        detailed_results.append({
            "file": fname,
            "stage1": "Accepted (Hen)",
            "stage1_prob": gate_prob,
            "stage2": pred_class,
            "confidence": conf,
            "probs": {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
        })
        
        if (idx + 1) % 50 == 0 or (idx + 1) == len(hen_files):
            print(f"  Processed {idx + 1} / {len(hen_files)} files...")

    # Metrics computation
    total_tested = len(hen_files)
    stage1_hen_recognition_rate = (stage1_passed / total_tested) * 100.0
    
    pred_counts = Counter(stage2_predictions)
    healthy_count = pred_counts.get("Healthy", 0)
    unhealthy_count = pred_counts.get("Unhealthy", 0) # False Positives for Avian Flu
    noise_count = pred_counts.get("Noise", 0)
    
    stage2_healthy_accuracy = (healthy_count / stage1_passed * 100.0) if stage1_passed > 0 else 0.0
    false_alarm_flu_rate = (unhealthy_count / stage1_passed * 100.0) if stage1_passed > 0 else 0.0
    avg_conf = np.mean(confidences) if confidences else 0.0
    
    print("\n" + "=" * 70)
    print("               NORMAL HEN TESTING BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Total Normal Hen Audio Files Tested:       {total_tested}")
    print(f"Stage 1 Hen Voice Filter Acceptance Rate:  {stage1_hen_recognition_rate:.2f}% ({stage1_passed} / {total_tested})")
    print(f"Stage 1 False Rejections (Non-Hen):        {stage1_rejected}")
    print("-" * 70)
    print("Stage 2 Deep ResNet Classification Breakdown (on Accepted Hen Audio):")
    print(f"  -> Correctly Classified as HEALTHY:      {healthy_count} ({stage2_healthy_accuracy:.2f}%)")
    print(f"  -> Misclassified as NOISE:               {noise_count} ({(noise_count / stage1_passed * 100.0) if stage1_passed else 0:.2f}%)")
    print(f"  -> FALSE ALARM: Avian Flu (Unhealthy):   {unhealthy_count} ({false_alarm_flu_rate:.2f}%)")
    print(f"Average Model Confidence on Healthy Hens:  {avg_conf * 100.0:.2f}%")
    print("=" * 70)
    
    # Check if goal is met
    if false_alarm_flu_rate < 5.0 and stage2_healthy_accuracy >= 90.0:
        print("[SUCCESS] Normal Hens are accurately classified as Healthy with near-zero false alarms!")
    else:
        print("[ANALYSIS] Reviewing misclassifications to optimize model boundaries...")

if __name__ == "__main__":
    run_large_hen_evaluation()
