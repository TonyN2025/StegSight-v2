import streamlit as st
import numpy as np
import soundfile as sf
from io import BytesIO
import struct
import wave

st.set_page_config(page_title="Audio Decode", page_icon="🔓", layout="centered")
st.title("🔓 Extract Text from Audio")

st.markdown("""
This page extracts text hidden in audio using **LSB steganography** (least significant bit of each sample).
- Recommended format: **PCM WAV (16-bit, 44.1kHz/48kHz)**
- Supports mono/stereo WAV. Message is extracted from LSBs.
""")

uploaded = st.file_uploader("Upload stego audio file (WAV)", type=["wav"])
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

def extract_bits_from_samples(samples: np.ndarray) -> list:
    lsb = (samples & 1).astype(np.uint8)
    return lsb.tolist()

def bits_to_message(bits: list) -> str:
    if len(bits) < 32:
        raise ValueError("Not enough bits for length prefix.")
    length_bytes = bytearray()
    for bidx in range(0, 32, 8):
        byte = 0
        for bitpos in range(8):
            bit = bits[bidx + bitpos]
            byte |= (bit & 1) << bitpos
        length_bytes.append(byte)
    length = struct.unpack("<I", bytes(length_bytes))[0]
    total_needed_bits = 32 + length * 8
    if len(bits) < total_needed_bits:
        raise ValueError("Not enough bits for declared message length.")
    msg_bytes = bytearray()
    for bstart in range(32, 32 + length * 8, 8):
        byte = 0
        for bitpos in range(8):
            bit = bits[bstart + bitpos]
            byte |= (bit & 1) << bitpos
        msg_bytes.append(byte)
    return msg_bytes.decode('utf-8', errors='replace')

if uploaded is not None:
    try:
        wav_bytes = uploaded.read()
        params, samples = wav_to_samples(wav_bytes)
        st.audio(wav_bytes, format="audio/wav")
        st.write(f"Channels: {params.nchannels}, Sample width: {params.sampwidth*8} bits, Frames: {params.nframes}, Sample rate: {params.framerate} Hz")
        st.write(f"Duration: {params.nframes / params.framerate:.2f} s, Total samples: {samples.size}")
        bits = extract_bits_from_samples(samples)
        st.write(f"Extracted bits: {len(bits)}")
        try:
            message_out = bits_to_message(bits)
            st.success("✅ Hidden message successfully decoded!")
            st.text_area("Decoded message", value=message_out, height=140)
        except Exception as e:
            st.error(f"Decoding error: {e}")
    except Exception as e:
        st.error(f"Error reading audio: {e}")
else:
    st.info("Upload a WAV file from the sidebar.")
