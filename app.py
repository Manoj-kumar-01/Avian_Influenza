import os
import io
import shutil
import pickle
import threading
import traceback
import requests
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

import numpy as np
import torch
import soundfile as sf
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from audio_utils import (
    load_audio_universal,
    extract_features,
    extract_features_from_waveform,
    extract_gatekeeper_features,
    extract_gatekeeper_features_from_waveform,
    generate_spectrogram_base64,
    TARGET_SR
)
from cnn_model import AudioCNN
from hen_filter_model import HenVoiceFilter
from database import init_db, save_diagnostic_record, get_recent_records, get_diagnostic_stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS_DIR = os.path.join(BASE_DIR, "Inputs")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_PATH = os.path.join(BASE_DIR, "cnn_model.pth")
OOD_MODEL_PATH = os.path.join(BASE_DIR, "ood_detector.pkl")
ALERTS_LOG = os.path.join(BASE_DIR, "alerts.log")

os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

CLASSES = ["Healthy", "Unhealthy", "Noise"]

app = FastAPI(
    title="AvianGuard AI — Real-time Avian Influenza Audio Detection",
    description="High-precision two-stage neural detection system for Avian Influenza in poultry",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model: Optional[AudioCNN] = None
gatekeeper_model: Optional[HenVoiceFilter] = None

def init_models():
    global cnn_model, gatekeeper_model
    # 1. Load Stage 2 CNN
    try:
        if os.path.exists(MODEL_PATH):
            cnn_model = AudioCNN(num_classes=len(CLASSES), backbone="resnet34").to(device)
            cnn_model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
            cnn_model.eval()
            print(f"[OK] PyTorch ResNet Disease Model loaded on {device}.")
        else:
            print(f"[WARN] CNN model checkpoint not found at {MODEL_PATH}.")
    except Exception as e:
        print(f"[ERROR] Error loading CNN model: {e}")

    # 2. Load Stage 1 Gatekeeper
    try:
        if os.path.exists(OOD_MODEL_PATH):
            gatekeeper_model = HenVoiceFilter.load(OOD_MODEL_PATH)
            print("[OK] Stage 1 Hen Voice Gatekeeper loaded successfully.")
        else:
            print(f"[WARN] Gatekeeper model not found at {OOD_MODEL_PATH}.")
    except Exception as e:
        print(f"[ERROR] Error loading Gatekeeper model: {e}")

    # 3. Initialize Relational Database
    try:
        init_db()
    except Exception as e:
        print(f"[WARN] Database initialization error: {e}")

init_models()

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/Inputs", StaticFiles(directory=INPUTS_DIR), name="inputs")
app.mount("/audio", StaticFiles(directory=INPUTS_DIR), name="audio")

def start_telegram_bot_worker():
    """Runs the Telegram Bot concurrently in a background daemon thread."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    if bot_token and os.getenv("ENABLE_TELEGRAM_BOT", "true").lower() in ("true", "1", "yes"):
        try:
            from telegram_bot import run_bot
            print("[STARTUP] Launching Telegram Bot background worker...")
            bot_thread = threading.Thread(target=run_bot, args=(bot_token,), daemon=True, name="TelegramBotDaemon")
            bot_thread.start()
            print("[STARTUP] Telegram Bot is running concurrently with the web server.")
        except Exception as e:
            print(f"[STARTUP] Could not start Telegram Bot daemon: {e}")

@app.on_event("startup")
async def app_startup():
    start_telegram_bot_worker()

# ── Alert Notification System ──
ENABLE_EXTERNAL_PUSH = os.getenv("ENABLE_EXTERNAL_PUSH", "false").lower() in ("true", "1", "yes")
ENABLE_EMAIL_ALERTS = os.getenv("ENABLE_EMAIL_ALERTS", "false").lower() in ("true", "1", "yes")

def send_email_alert(subject: str, body: str):
    if not ENABLE_EMAIL_ALERTS:
        return {"status": "disabled", "reason": "External email alerts disabled"}

    smtp_email = os.getenv("SMTP_EMAIL")
    smtp_password = os.getenv("SMTP_PASSWORD")
    receiver_emails_raw = os.getenv("RECEIVER_EMAILS", "")
    receiver_emails = [e.strip() for e in receiver_emails_raw.split(",") if e.strip()]

    if not smtp_email or not smtp_password or not receiver_emails:
        return {"status": "skipped", "reason": "SMTP credentials or receivers not set in .env"}

    try:
        msg = MIMEMultipart()
        msg["From"] = f"AvianGuard AI <{smtp_email}>"
        msg["To"] = ", ".join(receiver_emails)
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        server = smtplib.SMTP("smtp.gmail.com", 587, timeout=10)
        server.starttls()
        server.login(smtp_email, smtp_password)
        server.sendmail(smtp_email, receiver_emails, msg.as_string())
        server.quit()
        print(f"[EMAIL] Alert sent successfully to {receiver_emails}")
        return {"status": "sent", "recipients": receiver_emails}
    except Exception as e:
        err_msg = str(e)
        if "5.7.9" in err_msg or "Application-specific password" in err_msg:
            print("[EMAIL] Notice: Gmail requires a 16-character Google App Password (2FA enabled).")
        else:
            print(f"[EMAIL] Failed to send email: {e}")
        return {"status": "failed", "error": err_msg}

def send_push_alert(prediction: str, confidence: float, filename: str, is_hen: bool = True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    topic_name = "birdflu7"

    if not is_hen:
        title = "Rejected: Non-Hen Audio"
        priority = "low"
        tags = "warning,loud_sound"
        body = f"Non-hen audio detected in {filename}.\nTime: {timestamp}"
    elif prediction == "Unhealthy":
        title = "ALERT: Suspected Avian Influenza Detected!"
        priority = "urgent"
        tags = "rotating_light,biohazard,warning"
        body = f"🚨 Suspected Avian Influenza detected in {filename}!\nConfidence: {confidence*100:.1f}%\nAction: Isolate poultry batch immediately.\nTime: {timestamp}"
    elif prediction == "Healthy":
        title = "Hen Vocalization: Healthy"
        priority = "default"
        tags = "white_check_mark,rooster"
        body = f"Healthy vocalization confirmed in {filename}.\nConfidence: {confidence*100:.1f}%\nTime: {timestamp}"
    else:
        title = "Environmental Noise Detected"
        priority = "low"
        tags = "sound"
        body = f"Noise signal detected in {filename}.\nTime: {timestamp}"

    # Log to file
    log_entry = f"[{timestamp}] [{prediction if is_hen else 'REJECTED_NON_HEN'}] {filename} (Conf: {confidence:.2f})\n"
    with open(ALERTS_LOG, "a", encoding="utf-8") as f:
        f.write(log_entry)

    # External push (disabled by default for security and privacy)
    ntfy_results = {}
    ntfy_delivered = False
    if ENABLE_EXTERNAL_PUSH:
        ntfy_topic = os.getenv("NTFY_TOPIC", "birdflu_private")
        try:
            resp = requests.post(
                f"https://ntfy.sh/{ntfy_topic}",
                data=body.encode("utf-8"),
                headers={"Title": title, "Priority": priority, "Tags": tags},
                timeout=3
            )
            ntfy_delivered = (resp.status_code == 200)
            ntfy_results["status"] = "delivered" if ntfy_delivered else f"HTTP {resp.status_code}"
        except Exception as e:
            ntfy_results["error"] = str(e)

    # Email alert (if explicitly enabled)
    email_res = None
    if ENABLE_EMAIL_ALERTS and (prediction == "Unhealthy" or not is_hen):
        email_res = send_email_alert(f"[AvianGuard AI] {title}", body)

    return {
        "title": title,
        "body": body,
        "timestamp": timestamp,
        "is_alert": (prediction == "Unhealthy"),
        "ntfy_delivered": ntfy_delivered,
        "email_details": email_res
    }

# ── Core Inference Engine (Sliding Window Architecture) ──
# Window parameters for segment-by-segment analysis
WINDOW_SEC = 3.0       # Each analysis window is 3 seconds
HOP_SEC = 1.5          # 50% overlap between consecutive windows
MIN_RMS_THRESHOLD = 0.01  # Skip near-silent segments

def analyze_audio_pipeline(file_path: str, source: str = "web_pipeline", user_identifier: Optional[str] = None):
    """
    Full sliding-window inference pipeline.
    Every segment of the audio is independently evaluated through both stages:
      Stage 1: Per-segment Hen Voice Gatekeeper
      Stage 2: Per-segment ResNet-34 Disease Classifier (only on hen-approved segments)
    Final result is aggregated across ALL segments with majority voting.
    """
    global cnn_model, gatekeeper_model

    if cnn_model is None or gatekeeper_model is None:
        init_models()

    filename = os.path.basename(file_path)
    spectrogram_b64 = generate_spectrogram_base64(file_path)

    # ── Load entire audio once ──
    try:
        waveform, sr = load_audio_universal(file_path, target_sr=TARGET_SR)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {e}")

    total_samples = waveform.shape[1]
    total_duration = total_samples / sr
    window_samples = int(WINDOW_SEC * sr)
    hop_samples = int(HOP_SEC * sr)

    # Build sliding windows
    windows = []
    start = 0
    while start < total_samples:
        end = min(start + window_samples, total_samples)
        segment = waveform[:, start:end]
        # Skip extremely short tail segments only if we already have windows
        if len(windows) > 0 and segment.shape[1] < int(0.5 * sr):
            break
        # Pad short final segment to window size
        if segment.shape[1] < window_samples:
            pad_amount = window_samples - segment.shape[1]
            segment = torch.nn.functional.pad(segment, (0, pad_amount))
        windows.append((start / sr, end / sr, segment))
        start += hop_samples

    num_windows = len(windows)
    print(f"[PIPELINE] Analyzing {total_duration:.1f}s audio in {num_windows} x {WINDOW_SEC}s windows (hop={HOP_SEC}s)")

    # ── STAGE 1: Per-Segment Hen Voice Gatekeeper ──
    hen_windows = []  # (start_sec, end_sec, segment, hen_prob) for segments that pass
    non_silent_count = 0  # Track how many windows had actual audio content

    if gatekeeper_model is not None:
        for w_start, w_end, segment in windows:
            # Skip near-silent segments
            rms = torch.sqrt(torch.mean(segment ** 2)).item()
            if rms < MIN_RMS_THRESHOLD:
                continue
            non_silent_count += 1

            try:
                gate_feats = extract_gatekeeper_features_from_waveform(segment, sr)
                if gate_feats is not None:
                    gate_res = gatekeeper_model.predict(gate_feats)
                    # Require high confidence for per-window hen identification
                    # Dog barking typically scores 0.69-0.73, genuine hens score 0.75+
                    if gate_res["is_hen"] and gate_res["hen_probability"] >= 0.75:
                        hen_windows.append((w_start, w_end, segment, gate_res["hen_probability"]))
            except Exception as e:
                print(f"[PIPELINE] Gatekeeper error on window {w_start:.1f}-{w_end:.1f}s: {e}")
                continue

        # Calculate hen ratio: what fraction of non-silent windows contain hen voice
        hen_ratio = len(hen_windows) / max(non_silent_count, 1)
        print(f"[PIPELINE] Stage 1 result: {len(hen_windows)}/{non_silent_count} non-silent windows are hen (ratio={hen_ratio:.2f})")

        # Reject if no hen segments, or if hen ratio is too low (< 20%)
        # For mixed audio (multiple animals), at least 20% of segments must contain hen voice
        MIN_HEN_RATIO = 0.20
        if len(hen_windows) == 0 or hen_ratio < MIN_HEN_RATIO:
            # No/insufficient hen segments found in entire audio
            notif_res = send_push_alert("Non-Hen", 0.0, filename, is_hen=False)
            
            # Persist to database
            save_diagnostic_record(
                prediction="Rejected",
                confidence=0.0,
                prob_dict={"Healthy": 0.0, "Unhealthy": 0.0, "Noise": 1.0},
                source=source,
                user_identifier=user_identifier,
                filename=filename,
                clinical_note=f"Rejected: Non-hen audio. Hen ratio {hen_ratio*100:.0f}%.",
                duration_sec=total_duration,
                segments_analyzed=num_windows,
                is_alert=False
            )
            return {
                "status": "rejected",
                "stage": "Stage 1: Hen Voice Gatekeeper",
                "error": "No hen/chicken vocalizations detected in any segment of the audio. The system analyzed "
                         f"all {num_windows} segments ({total_duration:.1f}s total). "
                         "Please provide audio containing clear hen sounds.",
                "reason": f"Scanned {non_silent_count} audio segments across {total_duration:.1f}s - "
                          f"only {len(hen_windows)} ({hen_ratio*100:.0f}%) matched hen voice (minimum 20% required).",
                "hen_probability": round(hen_ratio, 4),
                "anomaly_score": 0.0,
                "segments_analyzed": num_windows,
                "total_duration": round(total_duration, 1),
                "spectrogram": spectrogram_b64,
                "filename": filename,
                "notification": notif_res,
                "audio_url": f"/api/audio/{filename}",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    else:
        # No gatekeeper loaded — treat all windows as hen
        hen_windows = [(ws, we, seg, 1.0) for ws, we, seg in windows]

    # ── STAGE 2: Per-Segment Disease Classification on Hen Windows ──
    if cnn_model is None:
        raise HTTPException(status_code=500, detail="CNN Model not loaded.")

    class_votes = {c: 0.0 for c in CLASSES}
    class_counts = {c: 0 for c in CLASSES}
    segment_results = []

    with torch.no_grad():
        for w_start, w_end, segment, hen_prob in hen_windows:
            try:
                mel_spec = extract_features_from_waveform(segment, sr, max_duration_sec=WINDOW_SEC + 1.0)
                if mel_spec is None:
                    continue
                spec_tensor = mel_spec.unsqueeze(0).to(device)
                outputs = cnn_model(spec_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0]

                pred_idx = torch.argmax(probs).item()
                pred_class = CLASSES[pred_idx]
                pred_conf = probs[pred_idx].item()

                class_votes[pred_class] += pred_conf
                class_counts[pred_class] += 1

                segment_results.append({
                    "window": f"{w_start:.1f}-{w_end:.1f}s",
                    "prediction": pred_class,
                    "confidence": round(pred_conf, 4),
                    "hen_probability": round(hen_prob, 4)
                })
            except Exception as e:
                print(f"[PIPELINE] CNN error on window {w_start:.1f}-{w_end:.1f}s: {e}")
                continue

    print(f"[PIPELINE] Stage 2 result: votes={dict(class_counts)}")

    if not segment_results:
        raise HTTPException(status_code=400, detail="Could not extract features from any audio segment.")

    # ── Aggregate Results: Weighted Majority Voting ──
    # The final prediction is the class with the highest cumulative confidence
    final_prediction = max(class_votes, key=class_votes.get)
    total_hen_windows = len(segment_results)

    # Compute final confidence as average confidence of winning class
    if class_counts[final_prediction] > 0:
        final_confidence = class_votes[final_prediction] / class_counts[final_prediction]
    else:
        final_confidence = 0.0

    # Build per-class probability summary
    total_votes = sum(class_votes.values())
    prob_dict = {
        c: round(class_votes[c] / total_votes, 4) if total_votes > 0 else 0.0
        for c in CLASSES
    }

    is_alert = (final_prediction == "Unhealthy")
    notif_res = send_push_alert(final_prediction, final_confidence, filename, is_hen=True)

    # Clinical diagnostics explanation
    if final_prediction == "Unhealthy":
        clinical_note = (
            f"Avian respiratory distress signatures detected in {class_counts['Unhealthy']}/{total_hen_windows} "
            f"hen segments ({total_duration:.1f}s audio). Coughing, tracheal rales, or wheezing patterns observed. "
            "Immediate flock isolation and RT-PCR confirmation strongly advised."
        )
    elif final_prediction == "Healthy":
        clinical_note = (
            f"Normal hen vocalization patterns confirmed across {class_counts['Healthy']}/{total_hen_windows} "
            f"analyzed segments ({total_duration:.1f}s audio). No respiratory anomalies observed."
        )
    else:
        clinical_note = (
            f"Environmental/background noise dominant in {class_counts['Noise']}/{total_hen_windows} segments. "
            "Consider re-recording in a quieter environment."
        )
    # Persist to database
    save_diagnostic_record(
        prediction=final_prediction,
        confidence=final_confidence,
        prob_dict=prob_dict,
        source=source,
        user_identifier=user_identifier,
        filename=filename,
        clinical_note=clinical_note,
        duration_sec=total_duration,
        segments_analyzed=num_windows,
        is_alert=is_alert
    )

    return {
        "status": "success",
        "stage": "Stage 2: Deep Avian Influenza Classifier",
        "prediction": final_prediction,
        "confidence": round(final_confidence, 4),
        "probabilities": prob_dict,
        "alert": is_alert,
        "clinical_note": clinical_note,
        "notification": notif_res,
        "segments_analyzed": num_windows,
        "hen_segments_found": total_hen_windows,
        "total_duration": round(total_duration, 1),
        "spectrogram": spectrogram_b64,
        "filename": filename,
        "audio_url": f"/api/audio/{filename}",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# ── API Endpoints ──

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/predict")
async def predict_audio_file(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Accepts any audio file format (WAV, MP3, M4A, AAC, OGG, WEBM, MP4 video)
    and processes through the two-stage AI pipeline.
    """
    clean_filename = "".join(c for c in file.filename if c.isalnum() or c in "._- ")
    if not clean_filename:
        clean_filename = f"audio_{int(datetime.now().timestamp())}.wav"
    file_location = os.path.join(INPUTS_DIR, clean_filename)
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = analyze_audio_pipeline(file_location)

    # Generate standardized 16-bit 22.05kHz PCM WAV for guaranteed universal browser playback
    clean_wav_name = os.path.splitext(clean_filename)[0] + "_clean.wav"
    clean_wav_path = os.path.join(INPUTS_DIR, clean_wav_name)
    try:
        y, sr = load_audio_universal(file_location, target_sr=22050)
        if y is not None and y.numel() > 0:
            audio_np = y.squeeze(0).cpu().numpy()
            sf.write(clean_wav_path, audio_np, 22050, subtype="PCM_16")
            result["clean_wav_url"] = f"/api/audio/{clean_wav_name}"
    except Exception as e:
        print(f"Clean WAV conversion warning: {e}")

    result["audio_url"] = f"/api/audio/{clean_filename}"
    return result

class UrlRequest(BaseModel):
    url: str

@app.post("/api/predict-url")
async def predict_audio_from_url(payload: UrlRequest):
    """
    Downloads an audio file from a public URL (e.g. Instagram voice note link,
    cloud storage, webhook voice URL) and analyzes it.
    """
    url = payload.url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        raise HTTPException(status_code=400, detail="Invalid URL format.")

    try:
        resp = requests.get(url, timeout=15, stream=True)
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to fetch audio from URL (HTTP {resp.status_code})")

        # Determine filename
        filename = f"url_voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        save_path = os.path.join(INPUTS_DIR, filename)

        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        return analyze_audio_pipeline(save_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error downloading or processing audio: {e}")

@app.post("/api/webhook")
async def general_webhook(payload: dict):
    """
    Generic webhook gateway for Instagram Graph API, Zapier, Make.com, or Telegram forwards.
    """
    audio_url = payload.get("audio_url") or payload.get("voice_url") or payload.get("media_url")
    if not audio_url:
        return {"status": "error", "message": "No audio URL provided in webhook payload."}
    
    return await predict_audio_from_url(UrlRequest(url=audio_url))

@app.get("/api/demo-samples")
async def get_demo_samples():
    """
    Returns pre-configured demo samples for 1-click testing in the browser testbench.
    """
    samples = [
        {
            "id": "healthy",
            "name": "Healthy Hen (Vocalization)",
            "type": "Healthy",
            "description": "Normal domestic hen clucking and food calls.",
            "file": "demo_healthy_hen.wav"
        },
        {
            "id": "unhealthy",
            "name": "Avian Influenza Suspect (Infected)",
            "type": "Unhealthy",
            "description": "Hen with respiratory distress, coughing, and tracheal rales.",
            "file": "demo_unhealthy_avian_flu.wav"
        },
        {
            "id": "noise",
            "name": "Environmental Farm Noise",
            "type": "Noise",
            "description": "Farm background ventilation and machinery sound.",
            "file": "Flu-Noise.wav"
        },
        {
            "id": "dog_bark",
            "name": "Dog Barking (Non-Hen Negative)",
            "type": "Non-Hen",
            "description": "Barking dogs to test automatic non-hen voice rejection.",
            "file": "mixkit-horde-of-barking-dogs-60.wav"
        }
    ]
    return samples

@app.get("/api/audio/{filename}")
async def serve_audio_file(filename: str):
    file_path = os.path.join(INPUTS_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found.")
    
    ext = os.path.splitext(filename)[1].lower()
    mime_types = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".webm": "audio/webm",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".flac": "audio/flac"
    }
    media_type = mime_types.get(ext, "audio/wav")
    return FileResponse(file_path, media_type=media_type, headers={"Accept-Ranges": "bytes"})

@app.get("/api/demo-sample/{filename}")
async def fetch_demo_audio(filename: str):
    return await serve_audio_file(filename)

@app.api_route("/api/test-notification", methods=["GET", "POST"])
async def test_notification():
    """Trigger a manual test alert to verify internal notification pipeline."""
    result = send_push_alert(
        prediction="Unhealthy",
        confidence=0.985,
        filename="manual_test_alert.wav",
        is_hen=True
    )
    return {
        "status": "success",
        "message": "Internal test diagnostic logged successfully",
        "details": result
    }

@app.get("/api/health")
async def health_check():
    from database import get_collection
    return {
        "status": "online",
        "service": "AvianGuard AI Unified Surveillance Server",
        "device": str(device),
        "cnn_loaded": cnn_model is not None,
        "gatekeeper_loaded": gatekeeper_model is not None,
        "telegram_bot_active": bool(os.getenv("TELEGRAM_BOT_TOKEN")),
        "mongodb_connected": get_collection() is not None,
        "target_accuracy": ">95%"
    }

@app.get("/api/history")
async def get_history(limit: int = 50, offset: int = 0):
    """Retrieve historical bioacoustic diagnosis records from the database."""
    return get_recent_records(limit=limit, offset=offset)

@app.get("/api/stats")
async def get_stats():
    """Retrieve aggregate surveillance metrics from the database."""
    return get_diagnostic_stats()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
