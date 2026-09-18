# 🐔 AvianGuard AI — Real-Time Avian Influenza Bioacoustic Surveillance

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Telegram Bot](https://img.shields.io/badge/Telegram_Bot-Integrated-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white)](https://telegram.org)

**AvianGuard AI** is a bioacoustic deep learning system engineered for early detection of **Avian Influenza (Bird Flu)** in poultry flocks through automated audio vocalization analysis. 

The system operates a high-precision two-stage neural pipeline:
1. **Stage 1 Hen Voice Gatekeeper:** An anomaly detection filter that validates incoming sound is from domestic poultry (rejecting dogs, humans, machinery, and farm ambiance).
2. **Stage 2 Deep ResNet Disease Classifier:** A fine-tuned bioacoustic ResNet-34 neural network classifying chicken vocalizations into **Healthy Poultry**, **Suspected Avian Influenza (respiratory distress, coughing, rales)**, and **Background Farm Noise** with **>95% accuracy**.

---

## 🏛️ System Architecture

```
┌───────────────────────────────┐
│     Audio Input Sources       │
│ • Telegram Voice Notes & OGG  │
│ • Web Microphone & MP3/WAV    │
│ • Farm Surveillance Microphones│
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│  Stage 1: Hen Gatekeeper      │
│  (Acoustic Moment Extraction) │ ──[Non-Hen Voice]──► ❌ Reject (Dog, Traffic, Human)
└───────────────┬───────────────┘
                │ [Domestic Hen Audio Confirmed]
                ▼
┌───────────────────────────────┐
│  Stage 2: ResNet Classifier   │
│  (128-Band Log-Mel Feature dB)│
└───────────────┬───────────────┘
                │
       ┌────────┴────────┬─────────────────┐
       ▼                 ▼                 ▼
   [Healthy]       [Unhealthy/Flu]      [Noise]
       │                 │                 │
       ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────┐
│                  Action & Alert Engine                  │
│ • Relational Database Logging (SQLite / PostgreSQL)     │
│ • Telegram Bot Instant Report & Viridis Heatmap Photo   │
│ • Real-Time Mobile Push Notification (ntfy topic)       │
│ • Automated Veterinary Alert Email (SMTP)               │
└─────────────────────────────────────────────────────────┘
```

---

## ✨ Features

- 🔬 **Two-Stage Inference Pipeline:** Filters out non-poultry environmental audio before disease diagnosis, eliminating false positives.
- 📊 **128-Band Log-Mel Bioacoustic Heatmap:** Generates scientific Viridis acoustic spectrograms showing frequency over time with dB energy intensity.
- 🤖 **Telegram Bot Interface:** Direct mobile diagnostics by sending voice notes or forwarding WhatsApp/Instagram audio to `@BirdFlu_bot`.
- 💾 **Relational Database Storage:** SQLAlchemy database recording all diagnostic history, timestamps, infection probabilities, duration, and clinical notes. Compatible with SQLite and cloud PostgreSQL.
- 🔔 **Multi-Channel Alert Integration:** Immediate push alerts via `ntfy.sh` (topic `birdflu7`) and automated SMTP email alerts when symptoms of Avian Influenza are flagged.
- 🐳 **Containerized & Cloud-Ready:** Pre-configured Docker, Docker Compose, Procfile, and Render Blueprint for 1-click deployment.

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/Manoj-kumar-01/Avian_Influenza.git
cd Avian_Influenza
```

### 2. Set Up Python Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env` and configure your credentials:
```bash
cp .env.example .env
```

Edit `.env`:
```env
# Telegram Bot Token (from @BotFather)
TELEGRAM_BOT_TOKEN=your_token_here

# Database URL (Default: local SQLite)
DATABASE_URL=sqlite:///avian_guard.db

# Email Alerts (Optional SMTP)
SMTP_EMAIL=your_email@gmail.com
SMTP_PASSWORD=your_app_password
RECEIVER_EMAILS=alerts@poultrycorp.com
```

### 4. Run the Web Server
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
Visit `http://localhost:8000/` in your browser.

### 5. Launch the Telegram Bot
```bash
python telegram_bot.py
```
Open Telegram and message your bot to start testing voice notes.

---

## 🐳 Docker Deployment

### Run with Docker Compose (Web API + Telegram Bot):
```bash
docker-compose up --build -d
```

### Run standalone Docker container:
```bash
docker build -t avianguard-ai .
docker run -d -p 8000:8000 --env-file .env avianguard-ai
```

---

## 🌐 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web application dashboard |
| `POST` | `/api/predict` | Upload and analyze any audio file (`.wav`, `.mp3`, `.ogg`, `.m4a`, `.webm`) |
| `POST` | `/api/predict-url` | Download and diagnose audio from an external URL |
| `GET` | `/api/history` | Query historical diagnostic records from the database |
| `GET` | `/api/stats` | Aggregate surveillance metrics (total tests, infection rate %, alert counts) |
| `GET` | `/api/demo-samples` | List canonical demo audio files for 1-click browser testing |
| `GET` | `/api/health` | Service health status and device (CPU/CUDA) verification |

---

## 🧠 Model Training & Dataset

To re-train or fine-tune models:
1. **Download Datasets:** `python download_datasets.py`
2. **Train Stage 1 Gatekeeper:** `python train_ood.py`
3. **Train Stage 2 ResNet Classifier:** `python train_cnn.py`
4. **Evaluate Model Metrics:** `python evaluate_models.py`

---

## 📄 License
This project is licensed under the MIT License — see the LICENSE file for details.
