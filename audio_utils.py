import os
import io
import base64
import warnings
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T

# Target sample rate for poultry bioacoustic analysis
TARGET_SR = 22050
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

def load_audio_universal(file_path: str, target_sr: int = TARGET_SR):
    """
    Universally loads ANY audio file (WAV, MP3, M4A, AAC, OGG, WEBM, MP4 video)
    using soundfile or falling back to imageio_ffmpeg.
    Returns:
        waveform: torch.Tensor of shape (1, num_samples), float32
        sample_rate: int
    """
    data = None
    sample_rate = None

    # Method 1: Try soundfile (natively handles WAV, FLAC, OGG, MP3)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data, sample_rate = sf.read(file_path)
    except Exception:
        data = None

    # Method 2: Fallback to imageio_ffmpeg (handles M4A, AAC, WEBM, MP4 video audio track)
    if data is None:
        try:
            import imageio_ffmpeg
            import subprocess

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe,
                "-i", file_path,
                "-f", "f32le",
                "-acodec", "pcm_f32le",
                "-ac", "1",
                "-ar", str(target_sr),
                "-"
            ]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            raw_audio, _ = proc.communicate()
            if proc.returncode == 0 and len(raw_audio) > 0:
                data = np.frombuffer(raw_audio, dtype=np.float32)
                sample_rate = target_sr
        except Exception as e:
            print(f"Universal loader ffmpeg error on {file_path}: {e}")

    if data is None or len(data) == 0:
        raise ValueError(f"Unable to decode audio format for file: {file_path}")

    # Ensure float32
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    # Convert stereo to mono
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Normalize audio levels
    max_amp = np.max(np.abs(data))
    if max_amp > 1.0:
        data = data / max_amp
    elif max_amp > 0:
        # Standardize volume floor
        data = data / (max_amp + 1e-6) * 0.9

    waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

    # Resample if sample_rate differs from target_sr
    if sample_rate != target_sr:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform, target_sr

def extract_features(file_path: str, max_duration_sec: float = 10.0):
    """
    Extract a normalized Log-Mel Spectrogram tensor for CNN classification.
    Returns:
        torch.Tensor: Shape (1, 128, time_steps)
    """
    try:
        waveform, sr = load_audio_universal(file_path)
        return extract_features_from_waveform(waveform, sr, max_duration_sec)
    except Exception as e:
        print(f"Error in extract_features for {file_path}: {e}")
        return None

def extract_features_from_waveform(waveform: torch.Tensor, sr: int, max_duration_sec: float = 10.0):
    """
    Extract Log-Mel Spectrogram from a waveform tensor (no file I/O).
    Args:
        waveform: torch.Tensor of shape (1, num_samples)
        sr: sample rate
    Returns:
        torch.Tensor: Shape (1, 128, time_steps)
    """
    # Truncate if excessively long
    max_samples = int(sr * max_duration_sec)
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]

    # Pad if shorter than 0.5s
    min_samples = int(sr * 0.5)
    if waveform.shape[1] < min_samples:
        pad_amount = min_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

    # Mel Spectrogram transform
    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=50.0,
        f_max=sr // 2
    )
    mel_spec = mel_transform(waveform)
    mel_spec_db = T.AmplitudeToDB(stype="power", top_db=80)(mel_spec)

    return mel_spec_db

def extract_gatekeeper_features(file_path: str):
    """
    Extract comprehensive acoustic features for Stage 1 Hen Voice Filter.
    Computes statistical moments across frequency bands, zero crossing rate,
    spectral centroid, and spectral energy distribution.
    Returns:
        np.ndarray: 1D feature vector
    """
    try:
        waveform, sr = load_audio_universal(file_path)
        return extract_gatekeeper_features_from_waveform(waveform, sr)
    except Exception as e:
        print(f"Error in extract_gatekeeper_features for {file_path}: {e}")
        return None

def extract_gatekeeper_features_from_waveform(waveform: torch.Tensor, sr: int):
    """
    Extract gatekeeper features from a waveform tensor (no file I/O).
    Args:
        waveform: torch.Tensor of shape (1, num_samples)
        sr: sample rate
    Returns:
        np.ndarray: 1D feature vector
    """
    wf_np = waveform.squeeze().numpy()

    # Extract 64-band Mel Spectrogram
    mel_transform = T.MelSpectrogram(
        sample_rate=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=64
    )
    mel_spec = mel_transform(waveform)
    mel_spec_db = T.AmplitudeToDB(top_db=80)(mel_spec).squeeze().numpy()

    # Zero crossing rate
    zcr = np.mean(np.abs(np.diff(np.sign(wf_np)))) / 2.0

    # RMS Energy
    rms = np.sqrt(np.mean(wf_np ** 2))

    # Mel band statistics
    mel_mean = np.mean(mel_spec_db, axis=1)
    mel_std = np.std(mel_spec_db, axis=1)
    mel_max = np.max(mel_spec_db, axis=1)
    mel_min = np.min(mel_spec_db, axis=1)

    # Spectral Centroid approximation across mel bands
    freqs = np.linspace(50, sr // 2, 64)
    weights = np.maximum(0, mel_spec_db.mean(axis=1) + 80)
    centroid = np.sum(freqs * weights) / (np.sum(weights) + 1e-8)

    features = np.concatenate([
        mel_mean,
        mel_std,
        mel_max,
        mel_min,
        np.array([zcr, rms, centroid / 10000.0])
    ])
    return features

def generate_spectrogram_base64(file_path: str):
    """
    Generates a high-quality viridis-style Log-Mel Spectrogram image (base64 PNG)
    using pure NumPy and Pillow for instant rendering with zero heavy dependencies.
    """
    try:
        from PIL import Image

        spec = extract_features(file_path)
        if spec is None:
            return None

        # Convert to numpy array of shape (mels, time)
        spec_np = spec.squeeze().cpu().numpy()

        # Flip vertically so low frequencies are at bottom
        spec_np = np.flipud(spec_np)

        # Normalize to [0, 255]
        s_min = np.percentile(spec_np, 5)
        s_max = np.percentile(spec_np, 99)
        norm_spec = np.clip((spec_np - s_min) / (s_max - s_min + 1e-6), 0, 1)

        # Generate scientific bioacoustic Viridis colormap
        # Deep purple/navy = background silence, Teal/Green = acoustic harmonics, Vivid Gold = hen coughing/vocal rales
        import matplotlib
        rgba = matplotlib.colormaps['viridis'](norm_spec)
        rgb_arr = (rgba[:, :, :3] * 255).astype(np.uint8)

        # Resize to attractive display resolution
        img = Image.fromarray(rgb_arr, mode="RGB")
        img = img.resize((560, 160), Image.Resampling.BILINEAR)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"
    except Exception as e:
        print(f"Spectrogram generation error: {e}")
        return None
