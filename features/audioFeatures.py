from __future__ import annotations
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# ------------- Existing Simple MFCC ----------
def extract_spectrogram_features(file_path: str) -> np.ndarray:
    """
    Load an audio file and extract simple spectrogram features.
    Returns MFCC mean values as a feature vector.
    """
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)

    # Extract MFCCs (Mel-Frequency Cepstral Coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Take mean across time for each MFCC coefficient
    mfccs_mean = np.mean(mfccs, axis=1)

    return mfccs_mean

# ---------- New: richer feature set for steganalysis ----------
def extract_mfcc_stats(file_path: str, n_mfcc: int = 13) -> dict:
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    feats = {}
    for name, M in [("mfcc", mfcc), ("delta", d1), ("delta2", d2)]:
        feats.update({f"{name}_mean_{i+1:02d}": float(np.mean(M[i])) for i in range(n_mfcc)})
        feats.update({f"{name}_std_{i+1:02d}": float(np.std(M[i]))  for i in range(n_mfcc)})
    return feats

def extract_fft_features(file_path: str, hf_cutoff: float = 8000.0) -> dict:
    """
    FFT-based global stats + high-frequency energy ratio (above hf_cutoff).
    """
    y, sr = librosa.load(file_path, sr=None)
    Y = np.fft.rfft(y)
    mag = np.abs(Y)
    freqs = np.fft.rfftfreq(len(y), d=1/sr)

    total_energy = float(np.sum(mag**2) + 1e-12)
    hf_mask = freqs >= min(hf_cutoff, sr/2 - 1)
    hf_energy = float(np.sum((mag[hf_mask]**2)) if np.any(hf_mask) else 0.0)

    return {
        "fft_mean": float(np.mean(mag)),
        "fft_std": float(np.std(mag)),
        "fft_energy": total_energy,
        "fft_hf_energy": hf_energy,
        "fft_hf_ratio": float(hf_energy / total_energy) if total_energy > 0 else 0.0,
    }

def extract_spectral_stats(file_path: str) -> dict:
    """
    Common spectral features used in audio forensics/steganalysis.
    """
    y, sr = librosa.load(file_path, sr=None)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) + 1e-12

    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.95)
    flatness = librosa.feature.spectral_flatness(S=S)
    zcr = librosa.feature.zero_crossing_rate(y)

    stats = lambda x: (float(np.mean(x)), float(np.std(x)))
    c_mean, c_std = stats(centroid)
    b_mean, b_std = stats(bandwidth)
    r_mean, r_std = stats(rolloff)
    f_mean, f_std = stats(flatness)
    z_mean, z_std = stats(zcr)

    return {
        "spec_centroid_mean": c_mean, "spec_centroid_std": c_std,
        "spec_bandwidth_mean": b_mean, "spec_bandwidth_std": b_std,
        "spec_rolloff95_mean": r_mean, "spec_rolloff95_std": r_std,
        "spec_flatness_mean": f_mean, "spec_flatness_std": f_std,
        "zcr_mean": z_mean, "zcr_std": z_std,
    }

# ---------- Visualisations ----------
def make_waveform_fig(file_path: str):
    y, sr = librosa.load(file_path, sr=None)
    fig, ax = plt.subplots(figsize=(8, 2.4))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig

def make_spectrogram_fig(file_path: str):
    y, sr = librosa.load(file_path, sr=None)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis="time", y_axis="log", ax=ax)
    ax.set_title("Spectrogram (log-frequency, dB)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    return fig

def make_scalogram_fig(file_path: str):
    """
    Optional CWT scalogram if PyWavelets is available.
    """
    try:
        import pywt
    except Exception:
        return None  # gracefully skip if not installed

    y, sr = librosa.load(file_path, sr=None)
    # Downsample for speed (optional)
    if sr > 22050:
        y = librosa.resample(y, orig_sr=sr, target_sr=22050)
        sr = 22050

    widths = np.arange(1, 128)
    cwtmatr, _ = pywt.cwt(y, widths, 'morl')
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(np.abs(cwtmatr), aspect='auto', cmap='viridis', origin='lower')
    ax.set_title("Scalogram (CWT, Morlet)")
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Wavelet scale")
    fig.tight_layout()
    return fig

# ---------- Simple heuristic suspicion score ----------
def suspicion_score(features: dict) -> tuple[float, dict]:
    """
    Transparent, hand-crafted score in [0,100] combining:
    - High-frequency energy ratio (fft_hf_ratio)
    - Spectral flatness (spec_flatness_mean)
    - MFCC variance proxy (mean std of MFCCs)
    This is NOT a trained model—replace later with ML.
    """
    # Pull with safe defaults
    hf_ratio = float(features.get("fft_hf_ratio", 0.0))
    flatness = float(features.get("spec_flatness_mean", 0.0))
    mfcc_std_vals = [v for k, v in features.items() if k.startswith("mfcc_std_")]
    mfcc_std_mean = float(np.mean(mfcc_std_vals)) if mfcc_std_vals else 0.0

    # Normalise roughly into [0,1] ranges (tunable)
    # - hf_ratio: typical speech/music 0.05–0.25; above ~0.35 may be suspicious
    n_hf = np.clip((hf_ratio - 0.10) / (0.35 - 0.10 + 1e-9), 0.0, 1.0)
    # - flatness: 0 (tonal) .. 1 (noise-like); high can indicate noise-like perturbations
    n_flat = np.clip((flatness - 0.15) / (0.5 - 0.15 + 1e-9), 0.0, 1.0)
    # - mfcc_std_mean: higher variability across frames may indicate perturbations
    n_mfccv = np.clip((mfcc_std_mean - 5.0) / (20.0 - 5.0 + 1e-9), 0.0, 1.0)

    score01 = 0.45 * n_hf + 0.35 * n_flat + 0.20 * n_mfccv
    score = float(100.0 * score01)

    explain = {
        "hf_ratio": hf_ratio, "flatness_mean": flatness, "mfcc_std_mean": mfcc_std_mean,
        "norm_hf": float(n_hf), "norm_flat": float(n_flat), "norm_mfcc_var": float(n_mfccv),
        "weights": {"hf": 0.45, "flat": 0.35, "mfcc_var": 0.20}
    }
    return score, explain

def extract_all_audio_features(file_path: str) -> dict:
    """
    Convenience aggregator for the detect page.
    """
    feats = {}
    feats.update(extract_fft_features(file_path))
    feats.update(extract_spectral_stats(file_path))
    feats.update(extract_mfcc_stats(file_path))
    return feats
