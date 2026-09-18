import time
import os
import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

INPUTS_DIR = "Inputs"
API_URL = "http://localhost:8000/api/predict"

class AudioFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".wav"):
            print(f"\n[Monitor] New audio file detected: {event.src_path}")
            self.process_file(event.src_path)

    def process_file(self, file_path):
        # Wait slightly to ensure file is fully written before reading
        time.sleep(1)
        
        print(f"[Monitor] Triggering AI inference for {os.path.basename(file_path)}...")
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'audio/wav')}
                response = requests.post(API_URL, files=files)
                
            if response.status_code == 200:
                data = response.json()
                print(f"[Monitor] Result: {data.get('prediction')} | Confidence: {data.get('probabilities', {}).get('Unhealthy', 0):.2f}")
                if data.get('alert'):
                    print(f"[Monitor] ⚠️ ALERT GENERATED ⚠️")
            else:
                print(f"[Monitor] API Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"[Monitor] Error processing file: {e}")

if __name__ == "__main__":
    os.makedirs(INPUTS_DIR, exist_ok=True)
    
    event_handler = AudioFileHandler()
    observer = Observer()
    observer.schedule(event_handler, INPUTS_DIR, recursive=False)
    observer.start()
    
    print(f"Watchdog monitoring started on directory: '{INPUTS_DIR}'")
    print("Waiting for new .wav files...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
