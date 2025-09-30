import streamlit as st
import numpy as np
import soundfile as sf
from numpy.fft import rfft

st.set_page_config(page_title="Audio Decode (Manual Fourier-QIM)", page_icon="🔓", layout="centered")
st.title("🔓 Manual Decode: Extract Hidden Text from Audio (Fourier/QIM)")

def bits_to_text(bits: np.ndarray) -> str:
    if len(bits) < 32:
        return ""
    nbytes = 0
    for i in range(32):
        nbytes = (nbytes << 1) | int(bits[i])
    total_bits = 32 + nbytes * 8
    if len(bits) < total_bits:
        return ""
    data_bits = bits[32:32+nbytes*8]
    b = np.packbits(data_bits)
    try:
        return b.tobytes().decode("utf-8", errors="replace")
    except Exception:
        return "<decode error>"

def pick_indices(sample_rate: int, nfft: int, low_hz: float, high_hz: float, step: int) -> np.ndarray:
    low_bin = int(np.ceil(low_hz * nfft / sample_rate))
    high_bin = int(np.floor(high_hz * nfft / sample_rate))
    low_bin = max(low_bin, 1)
    high_bin = min(high_bin, nfft // 2)
    if high_bin <= low_bin:
        return np.array([], dtype=int)
    idx = np.arange(low_bin, high_bin+1, step, dtype=int)
    return idx

def qim_decode(spec: np.ndarray, idx: np.ndarray, delta: float, max_bits: int=4096) -> np.ndarray:
    mags = np.abs(spec[idx])
    q = np.floor(mags / delta).astype(np.int64)
    bits = q & 1
    return bits[:max_bits]

uploaded = st.file_uploader("Upload stego audio file (WAV)", type=["wav"])

if uploaded is not None:
    try:
        audio, sr = sf.read(uploaded)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float64)
        st.audio(uploaded, format="audio/wav")
        nfft = len(audio)
        st.write(f"Sample rate: **{sr} Hz**, Duration: **{nfft/sr:.2f} s**, nfft: **{nfft} samples**")
    except Exception as e:
        st.error(f"Error reading audio: {e}")
        st.stop()

    st.subheader("Enter encoding parameters (from backup):")
    low_hz = st.number_input("Low cutoff frequency (Hz)", min_value=0.0, value=8000.0, step=100.0)
    high_hz = st.number_input("High cutoff frequency (Hz)", min_value=1000.0, value=18000.0, step=100.0)
    step_bins = st.number_input("Bin step interval", min_value=1, value=4, step=1)
    delta = st.number_input("Absolute delta", min_value=0.0, max_value=1.0, value=2.4374919021578433e-05, step=1e-8, format="%.9f")
    max_bits = st.number_input("Max bits to decode (capacity)", min_value=1, value=14858, step=1)
    
    spec = rfft(audio)
    idx = pick_indices(sr, nfft, low_hz, high_hz, int(step_bins))
    st.write(f"Detected capacity: {len(idx)} bins")

    bits = qim_decode(spec, idx, delta, max_bits=max_bits)
    st.write(f"Decoded bits: {len(bits)}")
    st.write("First 40 bits (decoded):", bits[:40])
    st.write("Prefix as int (decoded):", int("".join(str(b) for b in bits[:32]), 2))
    text = bits_to_text(bits)
    if text == "" or text == "<decode error>":
        st.warning("No valid message decoded. Please check parameters or try different delta/strength.")
    else:
        st.success("✅ Hidden message successfully decoded!")
        st.text_area("Decoded message", value=text, height=140)
else:
    st.info("Upload a WAV file from the sidebar.")
