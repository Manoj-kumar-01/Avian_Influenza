import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from audio_utils import extract_features
from cnn_model import AudioCNN
from audio_augmentations import AudioAugmentationPipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(BASE_DIR, "Inputs")
DATASET_PATH = os.path.join(INPUTS_DIR, "Chicken_Audio_Dataset")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "cnn_model.pth")

CLASSES = ["Healthy", "Unhealthy", "Noise"]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}

# Reproducibility seed
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)

class PrecomputedAudioDataset(Dataset):
    """
    In-memory cached audio dataset for maximum training throughput on GPU.
    """
    def __init__(self, items, is_training: bool = False):
        # items is list of (mel_tensor, label)
        self.items = items
        self.is_training = is_training
        self.augmentor = AudioAugmentationPipeline() if is_training else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        spec, label = self.items[idx]
        spec_tensor = spec.clone()

        if self.is_training and self.augmentor is not None:
            spec_tensor = self.augmentor.augment_spectrogram(spec_tensor)

        return spec_tensor, label

def collate_audio_batch(batch):
    features, labels = zip(*batch)
    # Shape is (1, 128, time)
    max_time = max(f.shape[2] for f in features)
    # Fix maximum temporal window to avoid huge pad memory
    max_time = min(max_time, 431) # ~10 seconds at 22050Hz/512 hop

    padded = []
    for f in features:
        if f.shape[2] > max_time:
            f_slice = f[:, :, :max_time]
        else:
            pad_len = max_time - f.shape[2]
            f_slice = torch.nn.functional.pad(f, (0, pad_len))
        padded.append(f_slice)

    return torch.stack(padded), torch.tensor(labels, dtype=torch.long)

def load_all_data():
    print(f"Scanning dataset from {DATASET_PATH}...")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset path does not exist: {DATASET_PATH}")

    all_samples = []
    for cls in CLASSES:
        folder = os.path.join(DATASET_PATH, cls)
        wav_files = glob.glob(os.path.join(folder, "*.wav"))
        print(f"Found {len(wav_files)} files for class: {cls}")
        for path in wav_files:
            spec = extract_features(path)
            if spec is not None:
                all_samples.append((spec, CLASS_TO_IDX[cls]))

    print(f"Total valid audio samples pre-extracted: {len(all_samples)}")
    return all_samples

def train_model():
    print("=" * 65)
    print("  Training Deep Avian Influenza ResNet Classifier (Target > 95%)")
    print("=" * 65)

    all_samples = load_all_data()
    if len(all_samples) == 0:
        print("[ERROR] No data found. Please run download_datasets.py first.")
        return

    features, labels = zip(*all_samples)
    
    # Stratified 80/20 train/test split
    train_idx, test_idx = train_test_split(
        range(len(all_samples)),
        test_size=0.20,
        stratify=labels,
        random_state=42
    )

    train_items = [all_samples[i] for i in train_idx]
    test_items = [all_samples[i] for i in test_idx]

    print(f"Train samples: {len(train_items)} | Test samples: {len(test_items)}")

    train_dataset = PrecomputedAudioDataset(train_items, is_training=True)
    test_dataset = PrecomputedAudioDataset(test_items, is_training=False)

    batch_size = 16
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_audio_batch
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_audio_batch
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model = AudioCNN(num_classes=len(CLASSES), backbone="resnet34").to(device)

    # Label smoothing regularizes confidence and improves generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    
    num_epochs = 40
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_test_acc = 0.0
    best_loss = float("inf")

    print("\nStarting model optimization...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()

        scheduler.step()
        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = running_loss / train_total

        # Validation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        test_acc = 100.0 * test_correct / test_total
        avg_test_loss = test_loss / test_total

        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1 or test_acc > best_test_acc:
            print(f"Epoch [{epoch+1:2d}/{num_epochs:2d}] | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.2f}%")

        if test_acc >= best_test_acc and (test_acc > 90.0 or avg_test_loss < best_loss):
            best_test_acc = test_acc
            best_loss = avg_test_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  [* BEST MODEL SAVED] Validation Accuracy: {best_test_acc:.2f}%")

    print(f"\nTraining Complete! Peak Testing Accuracy: {best_test_acc:.2f}%")

    # Load best model for final report
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=True))
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())

    final_acc = accuracy_score(all_targets, all_preds) * 100.0
    print("\n" + "=" * 65)
    print(f"  FINAL BEST EVALUATION REPORT (Accuracy: {final_acc:.2f}%)")
    print("=" * 65)
    print(classification_report(all_targets, all_preds, target_names=CLASSES))
    print("Confusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))

    return final_acc

if __name__ == "__main__":
    train_model()
