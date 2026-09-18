"""
AvianGuard AI — Official Telegram Bioacoustic Bot
Real-Time Avian Influenza (Bird Flu) Detection from Poultry Vocalizations.

Receives audio / voice notes from Telegram (mobile microphone, WhatsApp/Instagram forwards)
and executes a two-stage deep learning pipeline:
  1. Stage 1 Hen Voice Gatekeeper: Filters out non-hen sounds (humans, dogs, machinery).
  2. Stage 2 ResNet Classifier: High-precision Avian Influenza respiratory distress diagnosis (>95% accuracy).
Returns a structured veterinary diagnostic report and the 128-Band Log-Mel Bioacoustic Heatmap.
"""

import os
import io
import sys
import html
import time
import base64
import argparse
import tempfile
import requests
from dotenv import load_dotenv

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torchaudio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

load_dotenv()

# Import the core analysis pipeline from app.py
from app import analyze_audio_pipeline, cnn_model, gatekeeper_model, device, init_models
from audio_utils import load_audio_universal, extract_features_from_waveform, TARGET_SR

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")


def make_progress_bar(percentage: float, length: int = 10) -> str:
    """Renders a text progress bar e.g. [████████░░]."""
    clamped = max(0.0, min(1.0, percentage))
    filled = int(round(clamped * length))
    empty = length - filled
    return "█" * filled + "░" * empty


import threading

_heatmap_lock = threading.Lock()


def generate_bioacoustic_heatmap_png(file_path: str, prediction: str = "", confidence: float = 0.0) -> bytes:
    """
    Generates the scientific 128-Band Log-Mel Bioacoustic Heatmap image matching the website.
    Displays Viridis acoustic energy (dB), frequency (Hz), and time (seconds) axes.
    """
    try:
        waveform, sr = load_audio_universal(file_path, target_sr=TARGET_SR)
        total_duration = waveform.shape[1] / sr
        analysis_duration = min(total_duration, 30.0)

        spec = extract_features_from_waveform(waveform, sr, max_duration_sec=analysis_duration)
        if spec is None:
            return None

        spec_np = spec.squeeze().cpu().numpy()

        with _heatmap_lock:
            fig, ax = plt.subplots(figsize=(9, 3.6), dpi=160)
            fig.patch.set_facecolor("#0b1329")
            ax.set_facecolor("#0b1329")

            im = ax.imshow(
                spec_np,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                extent=[0, analysis_duration, 50, TARGET_SR // 2]
            )

            cbar = fig.colorbar(im, ax=ax, pad=0.02)
            cbar.set_label("Acoustic Energy (dB)", color="#94a3b8", fontsize=9)
            cbar.ax.yaxis.set_tick_params(color="#94a3b8")
            plt.setp(plt.get(cbar.ax.axes, "yticklabels"), color="#94a3b8", fontsize=8)

            # Standard ASCII title to ensure reliable font rendering across systems
            title_tag = ""
            if prediction == "Unhealthy":
                title_tag = f" — [ALERT: Suspected Avian Influenza | {confidence*100:.1f}%]"
            elif prediction == "Healthy":
                title_tag = f" — [HEALTHY POULTRY | {confidence*100:.1f}%]"
            elif prediction == "Noise":
                title_tag = f" — [ENVIRONMENTAL NOISE | {confidence*100:.1f}%]"
            elif prediction == "Rejected":
                title_tag = " — [NON-HEN AUDIO REJECTED]"

            ax.set_title(
                f"128-Band Log-Mel Bioacoustic Spectrogram{title_tag}",
                color="#f8fafc",
                fontsize=10.5,
                fontweight="bold",
                pad=10
            )
            ax.set_xlabel("Time (seconds)", color="#94a3b8", fontsize=9)
            ax.set_ylabel("Frequency (Hz)", color="#94a3b8", fontsize=9)
            ax.tick_params(colors="#94a3b8", labelsize=8)

            for spine in ax.spines.values():
                spine.set_color("#1e293b")

            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), edgecolor="none")
            plt.close(fig)
            return buf.getvalue()
    except Exception as e:
        print(f"[BOT] Heatmap generation failed: {e}")
        return None


def send_telegram_message(token: str, chat_id: int, text: str, parse_mode: str = "HTML"):
    """Sends a Telegram text message."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode, "disable_web_page_preview": True}
    try:
        resp = requests.post(url, json=payload, timeout=12)
        if not resp.json().get("ok"):
            # Fallback without parse_mode if entity parsing failed
            payload.pop("parse_mode", None)
            requests.post(url, json=payload, timeout=12)
    except Exception as e:
        print(f"[BOT] Failed to send Telegram message: {e}")


def send_telegram_photo(token: str, chat_id: int, image_bytes: bytes, caption: str = "", parse_mode: str = "HTML"):
    """
    Sends a Telegram photo with an optional HTML caption.
    Gracefully handles captions exceeding Telegram's 1024-character limit.
    """
    url = f"https://api.telegram.org/bot{token}/sendPhoto"

    # Telegram photo captions are strictly capped at 1024 characters
    if len(caption) > 1020:
        short_caption = caption[:1000] + "..."
        files = {"photo": ("spectrogram.png", image_bytes, "image/png")}
        data = {"chat_id": chat_id, "caption": short_caption, "parse_mode": parse_mode}
        try:
            requests.post(url, data=data, files=files, timeout=20)
        except Exception as e:
            print(f"[BOT] Failed to send photo: {e}")
        # Send full message separately
        send_telegram_message(token, chat_id, caption, parse_mode=parse_mode)
        return

    files = {"photo": ("spectrogram.png", image_bytes, "image/png")}
    data = {"chat_id": chat_id, "caption": caption, "parse_mode": parse_mode}
    try:
        resp = requests.post(url, data=data, files=files, timeout=20)
        if not resp.json().get("ok"):
            print(f"[BOT] Telegram sendPhoto returned error: {resp.text}")
            # Try without caption on failure, then send caption as text
            data.pop("caption", None)
            files = {"photo": ("spectrogram.png", image_bytes, "image/png")}
            requests.post(url, data=data, files=files, timeout=20)
            if caption:
                send_telegram_message(token, chat_id, caption, parse_mode=parse_mode)
    except Exception as e:
        print(f"[BOT] Failed to send Telegram photo: {e}")


def format_report_html(result: dict) -> str:
    """Formats the AI inference output matching the website's diagnostic structure."""
    status = result.get("status")

    # ── Case 1: Audio Rejected by Stage 1 Hen Gatekeeper ──
    if status == "rejected":
        hen_prob = result.get("hen_probability", 0.0) * 100.0
        reason = html.escape(str(result.get("reason", "Non-hen acoustic signature.")))
        duration = result.get("total_duration", 0.0)
        windows = result.get("segments_analyzed", 0)

        report = (
            "🐔 <b>AVIAN INFLUENZA DIAGNOSTIC REPORT</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "❌ <b>STATUS: NON-HEN AUDIO REJECTED</b>\n"
            "🛡️ <b>Filter:</b> <code>Stage 1 Bioacoustic Gatekeeper</code>\n"
            f"⏱ <b>Audio Duration:</b> <code>{duration:.1f}s</code> ({windows} segments scanned)\n\n"
            f"📊 <b>Hen Acoustic Match:</b> <code>{hen_prob:.1f}%</code> [{make_progress_bar(hen_prob / 100.0)}]\n\n"
            "📋 <b>Gatekeeper Diagnosis:</b>\n"
            f"<i>{reason}</i>\n\n"
            "💡 <i>Tip: Please send or forward an audio recording containing actual chicken/hen clucking or coughing sounds.</i>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "<i>AvianGuard AI • Real-Time Poultry Health Surveillance</i>"
        )
        return report

    # ── Case 2: Analyzed by Stage 2 ResNet Classifier ──
    pred = result.get("prediction", "Unknown")
    conf = result.get("confidence", 0.0) * 100.0
    probs = result.get("probabilities", {"Healthy": 0.0, "Unhealthy": 0.0, "Noise": 0.0})
    duration = result.get("total_duration", 0.0)
    total_segments = result.get("segments_analyzed", 0)
    hen_segments = result.get("hen_segments_found", 0)
    clinical_note = html.escape(str(result.get("clinical_note", "Diagnosis complete.")))

    p_healthy = probs.get("Healthy", 0.0)
    p_unhealthy = probs.get("Unhealthy", 0.0)
    p_noise = probs.get("Noise", 0.0)

    if pred == "Unhealthy":
        status_banner = "🚨 <b>DIAGNOSIS: SUSPECTED AVIAN INFLUENZA</b>"
        action_advice = (
            "\n⚠️ <b>IMMEDIATE ACTION PROTOCOL:</b>\n"
            "• Quarantine this poultry flock immediately\n"
            "• Restrict vehicle and personnel movement\n"
            "• Collect cloacal/tracheal swabs for RT-PCR confirmation\n"
            "• Notify local veterinary public health authorities\n"
        )
    elif pred == "Healthy":
        status_banner = "✅ <b>DIAGNOSIS: HEALTHY POULTRY</b>"
        action_advice = (
            "\n🌿 <b>VETERINARY STATUS:</b>\n"
            "• Acoustic dynamics match normal healthy flock vocalizations.\n"
            "• Routine bioacoustic surveillance recommended.\n"
        )
    else:
        status_banner = "🔊 <b>DIAGNOSIS: ENVIRONMENTAL NOISE</b>"
        action_advice = (
            "\n💡 <b>RECOMMENDATION:</b>\n"
            "• Background ambient noise dominates the recording.\n"
            "• Re-record closer to the birds in a quieter environment.\n"
        )

    report = (
        "🐔 <b>AVIAN INFLUENZA DIAGNOSTIC REPORT</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"{status_banner}\n"
        f"🎯 <b>Model Confidence:</b> <code>{conf:.1f}%</code>\n"
        f"⏱ <b>Analyzed Audio:</b> <code>{duration:.1f}s</code> ({hen_segments}/{total_segments} hen segments)\n\n"
        "📋 <b>STAGE 1 — Hen Gatekeeper:</b>\n"
        "✅ <code>Verified Domestic Poultry Vocalization</code>\n\n"
        "🔬 <b>STAGE 2 — Acoustic Probability Distribution:</b>\n"
        f"• 🟢 <b>Healthy Hen:</b> <code>{p_healthy*100:5.1f}%</code> [{make_progress_bar(p_healthy)}]\n"
        f"• 🔴 <b>Avian Flu:</b>   <code>{p_unhealthy*100:5.1f}%</code> [{make_progress_bar(p_unhealthy)}]\n"
        f"• ⚪ <b>Farm Noise:</b>  <code>{p_noise*100:5.1f}%</code> [{make_progress_bar(p_noise)}]\n\n"
        "🩺 <b>Veterinary Clinical Assessment:</b>\n"
        f"<i>{clinical_note}</i>"
        f"{action_advice}\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "<i>AvianGuard AI • Real-Time Poultry Health Surveillance</i>"
    )
    return report


def run_bot(token: str):
    """Main Telegram bot polling loop."""
    print("=" * 68)
    print("  🐔 AvianGuard AI — Telegram Bioacoustic Bot Online")
    print(f"  Device: {device} | Models Loaded: CNN & Hen Gatekeeper")
    print("  Listening for voice notes, audio files, and forwarded media...")
    print("=" * 68)

    offset = 0
    while True:
        try:
            updates_url = f"https://api.telegram.org/bot{token}/getUpdates?offset={offset}&timeout=30"
            resp = requests.get(updates_url, timeout=35).json()

            if not resp.get("ok"):
                time.sleep(2)
                continue

            for update in resp.get("result", []):
                offset = update["update_id"] + 1
                message = update.get("message") or update.get("channel_post")
                if not message:
                    continue

                chat_id = message.get("chat", {}).get("id")
                if not chat_id:
                    continue

                text_msg = message.get("text", "")

                # ── Handle /start Command ──
                if text_msg == "/start":
                    welcome_text = (
                        "🐔 <b>Welcome to AvianGuard AI!</b>\n\n"
                        "I am an AI-powered bioacoustic diagnostic bot that detects <b>Avian Influenza (Bird Flu)</b> "
                        "directly from chicken vocalizations and farm audio recordings.\n\n"
                        "🔍 <b>How to test:</b>\n"
                        "1. <b>Send or forward any voice note</b> recorded on your phone, WhatsApp, or Instagram.\n"
                        "2. <b>Or upload an audio file</b> (<code>.wav</code>, <code>.mp3</code>, <code>.m4a</code>, <code>.ogg</code>, <code>.aac</code>).\n\n"
                        "⚙️ <b>Two-Stage Neural Pipeline:</b>\n"
                        "• <b>Stage 1 Hen Gatekeeper:</b> Filters out non-hen sounds (humans, dogs, machinery).\n"
                        "• <b>Stage 2 Deep ResNet Classifier:</b> High-accuracy diagnosis (>95%) distinguishing Healthy vs. Infected vocalizations.\n"
                        "• <b>Spectrogram Visualizer:</b> Returns the 128-Band Log-Mel bioacoustic energy heatmap.\n\n"
                        "🎙️ <i>Send or forward an audio recording now to begin!</i>"
                    )
                    send_telegram_message(token, chat_id, welcome_text)
                    continue

                # ── Handle /help Command ──
                if text_msg == "/help":
                    help_text = (
                        "📖 <b>AvianGuard AI — User Guide</b>\n\n"
                        "• <b>Voice Recording:</b> Tap and hold the mic icon in Telegram to record chicken sounds directly, then release.\n"
                        "• <b>Forwarded Audio:</b> Forward voice notes from WhatsApp, Instagram, or file managers.\n"
                        "• <b>Output Report:</b> You will receive a complete diagnostic report with confidence %, probability breakdown, "
                        "and the Log-Mel bioacoustic spectrogram heatmap.\n\n"
                        "Commands:\n"
                        "/start — Welcome message & overview\n"
                        "/status — Check AI model & server health\n"
                        "/help — This instruction manual"
                    )
                    send_telegram_message(token, chat_id, help_text)
                    continue

                # ── Handle /status Command ──
                if text_msg == "/status":
                    status_text = (
                        "⚙️ <b>AvianGuard AI System Status:</b>\n\n"
                        f"• <b>Compute Device:</b> <code>{device}</code>\n"
                        f"• <b>Stage 1 Gatekeeper:</b> {'✅ Ready' if gatekeeper_model else '⚠️ Not Loaded'}\n"
                        f"• <b>Stage 2 ResNet Classifier:</b> {'✅ Ready (>95% accuracy)' if cnn_model else '⚠️ Not Loaded'}\n"
                        "• <b>Spectrogram Engine:</b> ✅ 128-Band Log-Mel Viridis\n"
                        "• <b>Status:</b> 🟢 Operational"
                    )
                    send_telegram_message(token, chat_id, status_text)
                    continue

                # ── Detect Audio / Voice / Video Media ──
                voice_data = (
                    message.get("voice")
                    or message.get("audio")
                    or message.get("video_note")
                    or message.get("video")
                )

                # Check document if mime type is audio or video
                doc = message.get("document")
                if doc:
                    mime = doc.get("mime_type", "").lower()
                    doc_name = doc.get("file_name", "").lower()
                    audio_exts = (".wav", ".mp3", ".ogg", ".oga", ".m4a", ".aac", ".flac", ".webm", ".mp4")
                    if "audio" in mime or "video" in mime or any(doc_name.endswith(ext) for ext in audio_exts):
                        voice_data = doc

                if not voice_data:
                    send_telegram_message(
                        token,
                        chat_id,
                        "🎙️ Please send or forward an <b>audio recording or voice note</b> to diagnose for Avian Influenza."
                    )
                    continue

                # Retrieve file information from Telegram
                file_id = voice_data.get("file_id")
                file_info = requests.get(f"https://api.telegram.org/bot{token}/getFile?file_id={file_id}").json()

                if not file_info.get("ok"):
                    send_telegram_message(token, chat_id, "⚠️ Failed to download audio from Telegram servers. Please try again.")
                    continue

                file_path_tg = file_info["result"]["file_path"]
                download_url = f"https://api.telegram.org/file/bot{token}/{file_path_tg}"

                # Send initial processing status
                send_telegram_message(token, chat_id, "🔄 <i>Analyzing bioacoustic patterns with Two-Stage AI...</i>")

                # Determine file extension matching the Telegram source
                ext = os.path.splitext(file_path_tg)[1].lower()
                if not ext:
                    ext = ".oga" if "voice" in message else ".wav"

                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                    audio_resp = requests.get(download_url, timeout=25)
                    tmp_file.write(audio_resp.content)
                    tmp_path = tmp_file.name

                try:
                    # 1. Run inference pipeline
                    result = analyze_audio_pipeline(tmp_path, source="telegram", user_identifier=str(chat_id))

                    # 2. Format detailed diagnostic text report
                    report_html = format_report_html(result)

                    pred = result.get("prediction", "Rejected" if result.get("status") == "rejected" else "Unknown")
                    conf = result.get("confidence", 0.0)

                    # 3. Generate the 128-Band Log-Mel Bioacoustic Heatmap (Viridis colormap)
                    heatmap_bytes = generate_bioacoustic_heatmap_png(tmp_path, prediction=pred, confidence=conf)

                    # Fallback to base64 spectrogram from pipeline if matplotlib heatmap failed
                    if not heatmap_bytes:
                        b64 = result.get("spectrogram")
                        if b64 and "," in b64:
                            heatmap_bytes = base64.b64decode(b64.split(",")[1])

                    # 4. Dispatch result to Telegram
                    if heatmap_bytes:
                        send_telegram_photo(token, chat_id, heatmap_bytes, caption=report_html, parse_mode="HTML")
                    else:
                        send_telegram_message(token, chat_id, report_html, parse_mode="HTML")

                except Exception as e:
                    print(f"[BOT] Pipeline processing error: {e}")
                    error_msg = (
                        "⚠️ <b>Diagnostic Processing Error:</b>\n"
                        f"<code>{html.escape(str(e))}</code>\n\n"
                        "Please ensure the audio file is not corrupt and contains clear domestic hen sounds."
                    )
                    send_telegram_message(token, chat_id, error_msg, parse_mode="HTML")
                finally:
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass

        except KeyboardInterrupt:
            print("\n[BOT] Telegram Bot stopped by user.")
            break
        except Exception as e:
            print(f"[BOT] Unexpected polling error: {e}")
            time.sleep(3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AvianGuard Telegram Bot")
    parser.add_argument("--token", type=str, default=BOT_TOKEN, help="Telegram Bot Token")
    args = parser.parse_args()

    active_token = args.token or os.getenv("TELEGRAM_BOT_TOKEN")

    if not active_token:
        print("\n" + "=" * 70)
        print("  ⚠️ TELEGRAM BOT TOKEN NOT CONFIGURED")
        print("=" * 70)
        print("To launch your Avian Influenza Telegram Bot in 1 minute:")
        print("  1. Open Telegram and search for: @BotFather")
        print("  2. Send '/newbot', give your bot a name (e.g., AvianGuard Flu Bot)")
        print("  3. Copy the HTTP API token provided by BotFather.")
        print("  4. Either:")
        print("     a) Add to .env: TELEGRAM_BOT_TOKEN=YOUR_TOKEN_HERE")
        print("     b) Run: python telegram_bot.py --token YOUR_TOKEN_HERE")
        print("=" * 70 + "\n")
    else:
        run_bot(active_token)
