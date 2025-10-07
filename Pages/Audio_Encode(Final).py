import streamlit as st
import numpy as np
import soundfile as sf
from io import BytesIO
import struct
import wave

st.set_page_config(page_title="Audio Encode", page_icon="🎛️", layout="centered")
st.title("🔒 Hide Text in Audio")

st.markdown("""
This page hides text inside audio using **LSB steganography** (least significant bit of each sample).
- Recommended format: **PCM WAV (16-bit, 44.1kHz/48kHz)**
- Supports mono/stereo WAV. Message is embedded in LSBs.
""")

# 기존 UI 폼 유지
uploaded = st.file_uploader("Upload an audio file (WAV, PCM 1–2 channels)", type=["wav"])
text = st.text_area("Enter the text message to hide", height=140, placeholder="Type your secret message here")

col1, col2, col3 = st.columns(3)
with col1:
    dummy_low_hz = st.number_input("Low cutoff frequency (Hz)", min_value=0.0, value=8000.0, step=100.0)
with col2:
    dummy_high_hz = st.number_input("High cutoff frequency (Hz)", min_value=1000.0, value=18000.0, step=100.0)
with col3:
    dummy_step_bins = st.number_input("Bin step interval", min_value=1, value=4, step=1)

scale_mode = st.radio("Strength scale", options=["Linear","Log10"], horizontal=True, index=1)
if scale_mode == "Linear":
    dummy_strength = st.slider("Quantization strength (delta scale)", min_value=1e-5, max_value=0.1, value=0.0015, step=1e-5)
else:
    dummy_exp = st.slider("log10(strength)", min_value=-5.0, max_value=-1.0, value=-3.0, step=0.01)
    dummy_strength = float(10**dummy_exp)
st.caption(f"Current strength ≈ {dummy_strength:.6f}")

def wav_to_samples(wav_bytes: bytes):
    bio = BytesIO(wav_bytes)
    with wave.open(bio, "rb") as w:
        params = w.getparams()
        nchannels, sampwidth, framerate, nframes, comptype, compname = params
        if sampwidth != 2:
            raise ValueError("Only 16-bit WAV (sampwidth=2) supported.")
        raw = w.readframes(nframes)
        samples = np.frombuffer(raw, dtype=np.int16)
        return params, samples.copy()

def samples_to_wav_bytes(params, samples: np.ndarray):
    bio = BytesIO()
    with wave.open(bio, "wb") as w:
        w.setparams(params)
        w.writeframes(samples.tobytes())
    return bio.getvalue()

def message_to_bits(msg: str) -> list:
    b = msg.encode('utf-8')
    length_prefix = struct.pack("<I", len(b))
    combined = length_prefix + b
    bits = []
    for byte in combined:
        for i in range(8):
            bits.append((byte >> i) & 1)
    return bits

def embed_message_in_samples(samples: np.ndarray, message: str) -> np.ndarray:
    bits = message_to_bits(message)
    capacity = samples.size
    if len(bits) > capacity:
        raise ValueError(f"Message too large. Need {len(bits)} bits but only {capacity} available.")
    new = samples.copy()
    for i, bit in enumerate(bits):
        new[i] = (new[i] & ~1) | bit
    return new

if uploaded is not None:
    try:
        wav_bytes = uploaded.read()
        params, samples = wav_to_samples(wav_bytes)
        st.audio(wav_bytes, format="audio/wav")
        st.write(f"Channels: {params.nchannels}, Sample width: {params.sampwidth*8} bits, Frames: {params.nframes}, Sample rate: {params.framerate} Hz")
        st.write(f"Duration: {params.nframes / params.framerate:.2f} s, Total samples: {samples.size}")
    except Exception as e:
        st.error(f"Error reading audio: {e}")
        st.stop()

    if text:
        try:
            new_samples = embed_message_in_samples(samples, text)
            stego_bytes = samples_to_wav_bytes(params._replace(nframes=len(new_samples)//params.nchannels), new_samples)
            st.success("✅ Secret message successfully hidden in audio!")
            st.download_button("📥 Download stego audio (WAV)", data=stego_bytes, file_name="stego.wav", mime="audio/wav")
            st.info(f"Message bits: {len(message_to_bits(text))}, Capacity: {samples.size} bits")
            with st.expander("Backup parameters", expanded=False):
                st.code({
                    "channels": params.nchannels,
                    "sample_rate": params.framerate,
                    "nframes": params.nframes,
                    "total_samples": samples.size,
                    "message_bits": len(message_to_bits(text))
                }, language="json")
        except Exception as e:
            st.error(f"Encoding error: {e}")
    else:
        st.warning("Please enter a text message to hide.")
else:
    st.info("Upload a WAV file from the sidebar.")



