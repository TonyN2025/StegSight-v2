import numpy as np
from PIL import Image
from scipy import stats, fft
import pandas as pd

def extract_basic_stats(img: Image.Image) -> list[float]:
    """
    Extract simple statistical features from an image.
    Converts to grayscale and computes:
    - mean
    - standard deviation
    - 25th, 50th, and 75th percentiles
    """
    arr = np.array(img.convert("L"))  # grayscale
    return [
        float(arr.mean()),
        float(arr.std()),
        float(np.percentile(arr, 25)),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 75)),
    ]

def extract_advanced_steganalysis_features(img: Image.Image) -> dict:
    """
    Extract comprehensive steganalysis features across multiple domains
    """
    gray_array = np.array(img.convert('L'))
    rgb_array = np.array(img)
    
    features = {}
    
    # Spatial domain features
    features['spatial_mean'] = float(np.mean(gray_array))
    features['spatial_std'] = float(np.std(gray_array))
    features['spatial_variance'] = float(np.var(gray_array))
    features['spatial_skewness'] = float(stats.skew(gray_array.flatten()))
    features['spatial_kurtosis'] = float(stats.kurtosis(gray_array.flatten()))
    
    # Histogram features
    hist, _ = np.histogram(gray_array, bins=256, range=(0, 255))
    features['hist_entropy'] = float(stats.entropy(hist))
    features['hist_energy'] = float(np.sum(hist**2))
    
    # Frequency domain features (FFT)
    fft_transform = fft.fft2(gray_array)
    fft_magnitude = np.abs(fft_transform)
    features['fft_mean'] = float(np.mean(fft_magnitude))
    features['fft_std'] = float(np.std(fft_magnitude))
    features['fft_energy'] = float(np.sum(fft_magnitude**2))
    
    # LSB analysis
    lsb_plane = gray_array & 1
    features['lsb_mean'] = float(np.mean(lsb_plane))
    features['lsb_std'] = float(np.std(lsb_plane))
    lsb_hist = np.histogram(lsb_plane, bins=2)[0]
    features['lsb_entropy'] = float(stats.entropy(lsb_hist))
    
    # LSB transitions
    diff = np.diff(lsb_plane.flatten())
    features['lsb_transitions'] = int(np.sum(np.abs(diff)))
    
    # Color channel features
    for i, channel in enumerate(['R', 'G', 'B']):
        channel_data = rgb_array[:, :, i]
        features[f'{channel}_mean'] = float(np.mean(channel_data))
        features[f'{channel}_std'] = float(np.std(channel_data))
        features[f'{channel}_skewness'] = float(stats.skew(channel_data.flatten()))
    
    return features

def assess_steganalysis_threat(features: dict) -> tuple:
    """
    Basic threat assessment based on steganalysis features
    """
    threat_score = 0
    indicators = []
    
    # LSB distribution check (should be close to 0.5 for natural images)
    if abs(features.get('lsb_mean', 0.5) - 0.5) > 0.02:
        threat_score += 1
        indicators.append("LSB distribution anomaly")
    
    # High LSB transitions (indicates potential data hiding)
    if features.get('lsb_transitions', 0) > 500000:  # Adjust threshold as needed
        threat_score += 1
        indicators.append("Excessive LSB transitions")
    
    # Abnormal frequency energy
    if features.get('fft_energy', 0) > 1e9:  # Adjust threshold as needed
        threat_score += 1
        indicators.append("Abnormal frequency energy")
    
    if threat_score == 0:
        return "safe", indicators
    elif threat_score == 1:
        return "suspicious", indicators
    else:
        return "dangerous", indicators