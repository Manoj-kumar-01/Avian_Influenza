import os
import shutil
import glob
from huggingface_hub import snapshot_download

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(BASE_DIR, "Inputs")
DATASET_DIR = os.path.join(INPUTS_DIR, "Chicken_Audio_Dataset")
NON_HEN_DIR = os.path.join(INPUTS_DIR, "Non_Hen_Dataset")

CATEGORIES = ["Healthy", "Unhealthy", "Noise"]

def download_chicken_dataset():
    print("=" * 65)
    print("  Fast Concurrent Download: Poultry Vocalization Dataset")
    print("=" * 65)

    os.makedirs(DATASET_DIR, exist_ok=True)
    for cat in CATEGORIES:
        os.makedirs(os.path.join(DATASET_DIR, cat), exist_ok=True)

    repo_id = "IceKhoffi/chicken-health-behavior-multimodal"
    
    print(f"Downloading files concurrently with snapshot_download from {repo_id}...")
    local_snapshot = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=["vocalization-dataset-for-early-disease-detection/*"],
        max_workers=16
    )
    print(f"Snapshot downloaded to cache: {local_snapshot}")
    
    # Copy files to clean structured directory
    source_base = os.path.join(local_snapshot, "vocalization-dataset-for-early-disease-detection")
    for cat in CATEGORIES:
        cat_src = os.path.join(source_base, cat)
        cat_dst = os.path.join(DATASET_DIR, cat)
        if os.path.exists(cat_src):
            files = glob.glob(os.path.join(cat_src, "*.wav"))
            for f in files:
                dest_f = os.path.join(cat_dst, os.path.basename(f))
                if not os.path.exists(dest_f):
                    shutil.copyfile(f, dest_f)
            print(f"  --> Copied {len(files)} {cat} files to {cat_dst}")
            
    print(f"\n[DONE] Chicken Dataset Ready at: {DATASET_DIR}")
    for cat in CATEGORIES:
        count = len(glob.glob(os.path.join(DATASET_DIR, cat, "*.wav")))
        print(f"  - {cat}: {count} files")

def setup_non_hen_dataset():
    print("\n" + "=" * 65)
    print("  Setting up Non-Hen Audio Dataset (For Hen Voice Gatekeeper)")
    print("=" * 65)
    
    os.makedirs(NON_HEN_DIR, exist_ok=True)
    
    # 1. Existing local non-hen samples in Inputs/
    existing_samples = [
        "mixkit-horde-of-barking-dogs-60.wav",
        "Flu-Noise.wav"
    ]
    for s in existing_samples:
        src = os.path.join(INPUTS_DIR, s)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(NON_HEN_DIR, s))
            
    import numpy as np
    import soundfile as sf
    from scipy.signal import lfilter
    
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # 1. Human speech fundamental simulation (vocal tract formants at 130Hz, 700Hz, 2200Hz)
    if not os.path.exists(os.path.join(NON_HEN_DIR, "synthetic_human_speech.wav")):
        speech_sim = (
            0.5 * np.sin(2 * np.pi * 130 * t) * (1 + 0.3 * np.sin(2 * np.pi * 5 * t)) +
            0.3 * np.sin(2 * np.pi * 700 * t) +
            0.2 * np.sin(2 * np.pi * 2200 * t)
        )
        speech_sim = (speech_sim / np.max(np.abs(speech_sim)) * 0.8).astype(np.float32)
        sf.write(os.path.join(NON_HEN_DIR, "synthetic_human_speech.wav"), speech_sim, sr)
        
    # 2. Fan and mechanical tractor hum (120Hz + harmonics)
    if not os.path.exists(os.path.join(NON_HEN_DIR, "synthetic_machinery_hum.wav")):
        hum = (
            0.6 * np.sin(2 * np.pi * 60 * t) +
            0.4 * np.sin(2 * np.pi * 120 * t) +
            0.2 * np.sin(2 * np.pi * 240 * t) +
            0.05 * np.random.randn(len(t))
        )
        hum = (hum / np.max(np.abs(hum)) * 0.8).astype(np.float32)
        sf.write(os.path.join(NON_HEN_DIR, "synthetic_machinery_hum.wav"), hum, sr)

    # 3. Wind / Pink ambient noise
    if not os.path.exists(os.path.join(NON_HEN_DIR, "synthetic_wind_ambient.wav")):
        noise = np.random.randn(len(t))
        b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
        a = [1, -2.494956002, 2.017265875, -0.522189400]
        pink = lfilter(b, a, noise)
        pink = (pink / np.max(np.abs(pink)) * 0.7).astype(np.float32)
        sf.write(os.path.join(NON_HEN_DIR, "synthetic_wind_ambient.wav"), pink, sr)
        
    # 4. Music chords simulation
    if not os.path.exists(os.path.join(NON_HEN_DIR, "synthetic_music_chord.wav")):
        chord = (
            0.4 * np.sin(2 * np.pi * 261.63 * t) + # C4
            0.4 * np.sin(2 * np.pi * 329.63 * t) + # E4
            0.4 * np.sin(2 * np.pi * 392.00 * t)   # G4
        )
        chord = (chord / np.max(np.abs(chord)) * 0.8).astype(np.float32)
        sf.write(os.path.join(NON_HEN_DIR, "synthetic_music_chord.wav"), chord, sr)

    # 5. Dog barking simulation (pitch sweeps 400Hz - 200Hz bursts)
    if not os.path.exists(os.path.join(NON_HEN_DIR, "synthetic_bark_bursts.wav")):
        bark = np.zeros_like(t)
        for burst_start in [0.2, 0.8, 1.5, 2.2]:
            idx_start = int(burst_start * sr)
            idx_end = min(len(t), idx_start + int(0.25 * sr))
            t_burst = t[idx_start:idx_end] - burst_start
            freq = 500 * np.exp(-12 * t_burst) + 150
            phase = 2 * np.pi * np.cumsum(freq) / sr
            envelope = np.sin(np.pi * np.linspace(0, 1, len(t_burst))) ** 2
            bark[idx_start:idx_end] += envelope * np.sin(phase)
        bark = (bark / (np.max(np.abs(bark)) + 1e-6) * 0.85).astype(np.float32)
        sf.write(os.path.join(NON_HEN_DIR, "synthetic_bark_bursts.wav"), bark, sr)

    non_hen_files = glob.glob(os.path.join(NON_HEN_DIR, "*.wav"))
    print(f"[DONE] Non-Hen Audio Dataset Ready: {len(non_hen_files)} files in {NON_HEN_DIR}")

if __name__ == "__main__":
    download_chicken_dataset()
    setup_non_hen_dataset()
