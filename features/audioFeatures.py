import numpy as np
import librosa

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
