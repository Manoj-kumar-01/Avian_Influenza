# Avian Influenza Detection - Project Documentation

## 1. Project Overview
The **Avian Influenza Detection** project is a machine learning and deep learning-based system designed to automatically monitor and detect signs of Avian Influenza (Bird Flu) through chicken vocalizations (audio). The system analyzes audio recordings from the environment, filters out anomalies (non-hen sounds), and classifies the vocalizations into specific categories, issuing real-time alerts if symptoms of illness are detected.

**Core Categories (Classes):**
1. `Healthy`
2. `Unhealthy` (Indicative of Avian Influenza or respiratory distress)
3. `Noise` (Environmental noise)

---

## 2. Dataset Details
**Name:** Chicken Audio Dataset  
**Structure:** 
The dataset consists of `.wav` audio files categorized into three distinct folders:
- `/Healthy/`
- `/Unhealthy/`
- `/Noise/`

**Audio Preprocessing Pipeline:**
- **Loading:** Audio is read using the `soundfile` library.
- **Normalization:** Converted to a standard `float32` format.
- **Mono Conversion:** Multi-channel (stereo) audio is averaged to a single channel (mono).
- **Resampling:** All audio is standardized to a **22,050 Hz** sample rate using `torchaudio`.
- **Feature Extraction:** 
  - Converted into a **Mel-Spectrogram**.
  - CNN uses 128 Mel bands.
  - OOD uses 64 Mel bands (and computes statistical aggregations).
  - Converted to a decibel (dB) scale using Amplitude-to-DB mapping.

---

## 3. Algorithms & Models Used
The project utilizes a hybrid approach, combining traditional machine learning and deep learning algorithms to ensure robust inference.

### 3.1. Out-of-Distribution (OOD) Detector - Isolation Forest
- **Purpose:** Acts as a "Gatekeeper" to ensure the uploaded audio is actually a hen's vocalization and not random human or environmental noise.
- **Algorithm:** `IsolationForest` (from `scikit-learn`).
- **Features:** Computes a compact feature vector by taking the mean, standard deviation, max, and min per mel band.
- **Hyperparameters:** `n_estimators=200`, `contamination=0.05`.
- **Training Script:** `train_ood.py`
- **Output:** Saves `ood_detector.pkl`.

### 3.2. Convolutional Neural Network (CNN) - ResNet50
- **Purpose:** The primary classifier to distinguish between Healthy, Unhealthy, and Noise.
- **Algorithm:** Deep CNN based on the **ResNet50** architecture (PyTorch).
- **Modifications:** 
  - The first convolutional layer (`conv1`) is modified to accept 1-channel input (Mel-spectrogram).
  - The final fully connected layer (`fc`) is replaced with a Dropout layer (`p=0.5`) and a Linear layer outputting 3 classes.
- **Training Techniques:**
  - **Data Augmentation:** SpecAugment (Time Masking and Frequency Masking).
  - **Optimizer:** Adam Optimizer (`lr=0.001`).
  - **Learning Rate Scheduler:** Cosine Annealing (`T_max=50`).
  - **Loss Function:** Cross-Entropy Loss.
- **Training Script:** `train_cnn.py` & `cnn_model.py`
- **Output:** Saves `cnn_model.pth`.

### 3.3. Random Forest Classifier (Baseline Model)
- **Purpose:** An alternative/baseline machine learning model for classification.
- **Algorithm:** `RandomForestClassifier` (from `scikit-learn`).
- **Hyperparameters:** `n_estimators=100`, `random_state=42`.
- **Training Script:** `model_training.py`
- **Output:** Saves `model.pkl`.

---

## 4. Model Performance Accuracies
*(Note: Accuracies depend on the specific dataset splits and sizes used during local execution. The exact values generated during your run can be filled in below.)*

### CNN Model (ResNet50)
- **Training Accuracy:** `[Insert Final Training Accuracy, e.g., 95.5%]`
- **Testing/Validation Accuracy:** `[Insert Final Testing Accuracy, e.g., 92.3%]`
- **Epochs:** 50

### Random Forest Model
- **Training/Testing Accuracy:** `[Insert Final Accuracy, e.g., 88.0%]`

### OOD Detector (Isolation Forest)
- **Inlier Detection Rate:** `~95%` (Based on 0.05 contamination rate).

---

## 5. End-to-End System Architecture (Inference)
The deployment of the models is handled by a **FastAPI** backend (`app.py`), providing a REST API endpoint and a web frontend.

### The Inference Pipeline (`/api/predict`):
1. **Audio Upload:** The user uploads a `.wav` file through the frontend.
2. **OOD Check:** The system extracts OOD features and passes them through the Isolation Forest.
   - *If rejected (-1)*: Returns an error asking for valid audio (non-hen audio).
   - *If accepted (1)*: Proceeds to the CNN.
3. **CNN Classification:** The system extracts full Mel-spectrogram features and feeds them to the ResNet50 model.
4. **Probability Calculation:** Applies Softmax to get confidence scores for Healthy, Unhealthy, and Noise.
5. **Alert System Integration:** 
   - A Push Notification is triggered using **ntfy.sh** (Topic: `birdflu7`).
   - If `Unhealthy` is detected, it flags an `Urgent` priority alert (🚨).
   - Logs the transaction with a timestamp in `alerts.log`.
6. **Response:** Returns the prediction, probabilities, and alert status back to the user.

---

## 6. How to Run the Project
1. **Environment Setup:** Create a virtual environment and install dependencies (`torch`, `torchaudio`, `fastapi`, `scikit-learn`, `soundfile`, etc.).
2. **Train Models (Optional if pre-trained models exist):**
   - Run `python train_ood.py` to generate `ood_detector.pkl`.
   - Run `python train_cnn.py` to generate `cnn_model.pth`.
3. **Start the API Server:**
   - Run `python app.py` (which internally triggers `uvicorn`).
   - The web interface will be accessible at `http://localhost:8000/`.
4. **Launch the Telegram Bot:**
   - Add `TELEGRAM_BOT_TOKEN=YOUR_TOKEN` in `.env` (or pass via `--token`).
   - Run `python telegram_bot.py`.
   - Send any voice note or audio file to the bot for real-time diagnosis, probability breakdown, and the 128-Band Log-Mel bioacoustic heatmap.
