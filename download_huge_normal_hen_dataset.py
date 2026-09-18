import os
import glob
import urllib.request
import json
import csv
import io
import torch
import torchaudio
import soundfile as sf
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(BASE_DIR, "Inputs")
TEST_HEN_DIR = os.path.join(INPUTS_DIR, "Huge_Normal_Hen_Dataset")
os.makedirs(TEST_HEN_DIR, exist_ok=True)

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

def download_file(url, save_path):
    if os.path.exists(save_path) and os.path.getsize(save_path) > 1000:
        return True
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=30) as resp, open(save_path, 'wb') as out_f:
            out_f.write(resp.read())
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

# 1. Download ESC-50 'hen' files
def download_esc50_hens():
    print("\n[1/3] Ingesting ESC-50 'hen' recordings...")
    meta_url = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
    req = urllib.request.Request(meta_url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read().decode('utf-8')
    
    reader = csv.DictReader(io.StringIO(content))
    hen_rows = [r for r in reader if r.get('category') == 'hen']
    print(f"Found {len(hen_rows)} 'hen' rows in ESC-50.")
    
    count = 0
    for r in hen_rows:
        fname = r['filename']
        raw_url = f"https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/{fname}"
        dst = os.path.join(TEST_HEN_DIR, f"esc50_hen_{fname}")
        if download_file(raw_url, dst):
            count += 1
    print(f"Successfully downloaded {count} ESC-50 hen files.")

# 2. Download YashNita Tavuk files
def download_yashnita_tavuk():
    print("\n[2/3] Ingesting YashNita Animal-Sound-Dataset 'Tavuk' (Chicken)...")
    api_url = "https://api.github.com/repos/YashNita/Animal-Sound-Dataset/contents/Tavuk"
    req = urllib.request.Request(api_url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            items = json.loads(resp.read().decode('utf-8'))
        count = 0
        for it in items:
            if it['name'].endswith('.wav'):
                raw_url = it['download_url']
                dst = os.path.join(TEST_HEN_DIR, f"yashnita_{it['name']}")
                if download_file(raw_url, dst):
                    count += 1
        print(f"Successfully downloaded {count} YashNita chicken files.")
    except Exception as e:
        print(f"Error fetching YashNita: {e}")

# 3. Download & Segment zebular13 ChickenLanguageDataset
def download_and_slice_chicken_language():
    print("\n[3/3] Ingesting & Slicing ChickenLanguageDataset (Continuous field recordings)...")
    temp_download_dir = os.path.join(INPUTS_DIR, "Chicken_Temp_Long")
    os.makedirs(temp_download_dir, exist_ok=True)
    
    subdirs = [
        'longer_segments/eating ',
        'longer_segments/greeting',
        'longer_segments/where_is_everyone',
        'longer_segments/tidbitting_hen',
        'longer_segments/ground_alarm',
        'single_vocalizations/eating',
        'single_vocalizations/greeting',
        'single_vocalizations/where_is_everyone',
        'single_vocalizations/disturbed_in_nest_box',
        'single_vocalizations/hungry',
        'single_vocalizations/tidbitting_hen',
    ]
    
    slice_count = 0
    direct_count = 0
    
    for sub in subdirs:
        s_url = sub.replace(' ', '%20')
        api_url = f"https://api.github.com/repos/zebular13/ChickenLanguageDataset/contents/{s_url}"
        req = urllib.request.Request(api_url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                items = json.loads(resp.read().decode('utf-8'))
        except Exception as e:
            print(f"Error loading {sub}: {e}")
            continue
            
        for it in items:
            name = it['name']
            if not name.endswith(('.wav', '.mp3')):
                continue
            download_url = it['download_url']
            clean_name = "".join(c if c.isalnum() else "_" for c in name[:-4])
            temp_path = os.path.join(temp_download_dir, f"{clean_name}{os.path.splitext(name)[1]}")
            
            if download_file(download_url, temp_path):
                try:
                    wav, sr = torchaudio.load(temp_path)
                    if wav.shape[0] > 1:
                        wav = torch.mean(wav, dim=0, keepdim=True)
                    if sr != 22050:
                        wav = torchaudio.functional.resample(wav, sr, 22050)
                        sr = 22050
                    
                    y = wav.squeeze().numpy()
                    duration = len(y) / sr
                    
                    if duration <= 4.0:
                        dst = os.path.join(TEST_HEN_DIR, f"cld_{clean_name}.wav")
                        sf.write(dst, y, sr)
                        direct_count += 1
                    else:
                        seg_len = int(3.0 * sr)
                        hop_len = int(2.5 * sr)
                        idx = 0
                        for start in range(0, len(y) - seg_len, hop_len):
                            chunk = y[start : start + seg_len]
                            rms = np.sqrt(np.mean(chunk**2))
                            if rms > 0.005: # filter out pure silence
                                dst = os.path.join(TEST_HEN_DIR, f"cld_{clean_name}_seg{idx:03d}.wav")
                                sf.write(dst, chunk, sr)
                                slice_count += 1
                                idx += 1
                except Exception as e:
                    print(f"Error processing {name}: {e}")
                    
    print(f"Ingested {direct_count} short vocalizations + generated {slice_count} 3s continuous hen segments.")

if __name__ == "__main__":
    download_esc50_hens()
    download_yashnita_tavuk()
    download_and_slice_chicken_language()
    
    total_files = glob.glob(os.path.join(TEST_HEN_DIR, "*.wav"))
    print("\n" + "=" * 60)
    print(f"TOTAL NORMAL HEN DATASET SIZE ASSEMBLED: {len(total_files)} AUDIO FILES")
    print(f"Location: {TEST_HEN_DIR}")
    print("=" * 60)
