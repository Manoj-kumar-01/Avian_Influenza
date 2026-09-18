import os
import urllib.request
import json
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(BASE_DIR, "Inputs")
CHICKEN_DIR = os.path.join(INPUTS_DIR, "Chicken_Audio_Dataset")
HEALTHY_DIR = os.path.join(CHICKEN_DIR, "Healthy")
NON_HEN_DIR = os.path.join(INPUTS_DIR, "Non_Hen_Dataset")

os.makedirs(HEALTHY_DIR, exist_ok=True)
os.makedirs(NON_HEN_DIR, exist_ok=True)

HEADERS = {'User-Agent': 'Mozilla/5.0'}

def fetch_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode('utf-8'))

def download_chicken_language_dataset():
    print("[*] Gathering additional poultry vocalizations from ChickenLanguageDataset...")
    base_api = "https://api.github.com/repos/zebular13/ChickenLanguageDataset/contents/single_vocalizations"
    try:
        categories = fetch_json(base_api)
        downloaded = 0
        for cat in categories:
            if cat['type'] == 'dir':
                cat_name = cat['name']
                cat_url = f"{base_api}/{cat_name}"
                try:
                    files = fetch_json(cat_url)
                    for f in files:
                        if f['name'].endswith('.wav'):
                            dest = os.path.join(HEALTHY_DIR, f"cld_{cat_name}_{f['name']}")
                            if not os.path.exists(dest):
                                urllib.request.urlretrieve(f['download_url'], dest)
                                downloaded += 1
                except Exception as e:
                    print(f"[-] Error fetching category {cat_name}: {e}")
        print(f"[+] Downloaded {downloaded} additional authentic hen vocalizations into Healthy/")
    except Exception as e:
        print(f"[-] ChickenLanguageDataset error: {e}")

def download_esc50_samples():
    print("[*] Ingesting ESC-50 Bioacoustic & Negative Control Dataset...")
    csv_url = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/meta/esc50.csv"
    req = urllib.request.Request(csv_url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            lines = resp.read().decode('utf-8').splitlines()
        
        reader = csv.DictReader(lines)
        rows = list(reader)

        # 1. Roosters -> Poultry Healthy
        rooster_rows = [r for r in rows if r['category'] == 'rooster']
        rooster_count = 0
        for r in rooster_rows:
            fname = r['filename']
            url = f"https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/{fname}"
            dest = os.path.join(HEALTHY_DIR, f"esc50_rooster_{fname}")
            if not os.path.exists(dest):
                urllib.request.urlretrieve(url, dest)
                rooster_count += 1
        print(f"[+] Ingested {rooster_count} rooster vocalizations into Healthy/")

        # 2. Non-Hen Negative Controls (Human cough, sneeze, wild birds, pets, farm machinery)
        negative_categories = [
            'coughing', 'sneezing', 'breathing', 'crying_baby', 
            'dog', 'cat', 'cow', 'pig', 'chirping_birds', 
            'chainsaw', 'car_horn', 'rain', 'wind'
        ]
        non_hen_rows = [r for r in rows if r['category'] in negative_categories]
        # Download up to 3 per category for balanced representation
        cat_counts = {c: 0 for c in negative_categories}
        neg_count = 0
        for r in non_hen_rows:
            cat = r['category']
            if cat_counts[cat] < 3:
                fname = r['filename']
                url = f"https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/{fname}"
                dest = os.path.join(NON_HEN_DIR, f"esc50_{cat}_{fname}")
                if not os.path.exists(dest):
                    urllib.request.urlretrieve(url, dest)
                    neg_count += 1
                cat_counts[cat] += 1
        print(f"[+] Ingested {neg_count} diverse non-hen audio controls into Non_Hen_Dataset/")

    except Exception as e:
        print(f"[-] ESC-50 download error: {e}")

if __name__ == "__main__":
    download_chicken_language_dataset()
    download_esc50_samples()
    print("\n[✓] Extended dataset collection complete!")
    print(f"    Total Healthy files: {len(os.listdir(HEALTHY_DIR))}")
    print(f"    Total Non-Hen files: {len(os.listdir(NON_HEN_DIR))}")
