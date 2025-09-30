import streamlit as st
import numpy as np
import soundfile as sf
from io import BytesIO
from numpy.fft import rfft, irfft

st.set_page_config(page_title="Audio Encode (Fourier-QIM)", page_icon="🎛️", layout="centered")
st.title("🔒 Hide Text in Audio (Fourier/QIM)")

st.markdown("""
This page hides text inside audio using **Quantization Index Modulation (QIM)** in the **frequency domain (Fourier)**.
- Recommended format: **PCM WAV (16-bit, 44.1kHz/48kHz)**
- Stereo will be converted to **mono** automatically.
""")

def text_to_bits(s: str) -> np.ndarray:
    b = s.encode("utf-8")
    bits = np.unpackbits(np.frombuffer(b, dtype=np.uint8))
    return bits.astype(np.uint8)

def add_length_prefix(bits: np.ndarray) -> np.ndarray:
    nbytes = (len(bits) + 7) // 8
    prefix = np.array([(nbytes >> i) & 1 for i in range(31, -1, -1)], dtype=np.uint8)
    return np.concatenate([prefix, bits])

def pick_indices(sample_rate: int, nfft: int, low_hz: float, high_hz: float, step: int) -> np.ndarray:
    low_bin = int(np.ceil(low_hz * nfft / sample_rate))
    high_bin = int(np.floor(high_hz * nfft / sample_rate))
    low_bin = max(low_bin, 1)
    high_bin = min(high_bin, nfft // 2)
    if high_bin <= low_bin:
        return np.array([], dtype=int)
    idx = np.arange(low_bin, high_bin+1, step, dtype=int)
    return idx

def apply_qim_encode(spec: np.ndarray, idx: np.ndarray, bits: np.ndarray, delta: float) -> None:
    mags = np.abs(spec[idx])
    phases = np.angle(spec[idx])
    eps = 1e-12
    mags = np.maximum(mags, eps)
    q = np.floor(mags / delta).astype(np.int64)
    parity = (q & 1)
    target_parity = bits[: len(idx)]
    adj = (target_parity != parity).astype(np.int64)
    q_new = q + np.where(adj == 1, 1, 0)
    mags_new = (q_new + 0.5) * delta
    spec[idx] = mags_new * np.exp(1j * phases)

def estimate_delta(spec: np.ndarray, idx: np.ndarray, strength: float) -> float:
    if len(idx) == 0:
        return 0.0
    mags = np.abs(spec[idx])
    base = np.percentile(mags, 70)
    delta = max(base * strength, 1e-9)
    return float(delta)

uploaded = st.file_uploader("Upload an audio file (WAV, PCM 1–2 channels)", type=["wav"])
text = st.text_area("Enter the text message to hide", height=140, placeholder="Type your secret message here")

col1, col2, col3 = st.columns(3)
with col1:
    low_hz = st.number_input("Low cutoff frequency (Hz)", min_value=0.0, value=8000.0, step=100.0, help="Frequencies below this will not be modified.")
with col2:
    high_hz = st.number_input("High cutoff frequency (Hz)", min_value=1000.0, value=18000.0, step=100.0, help="Frequencies above this will not be modified.")
with col3:
    step_bins = st.number_input("Bin step interval", min_value=1, value=4, step=1, help="Spacing between frequency bins. Larger values = less distortion but lower capacity.")

scale_mode = st.radio("Strength scale", options=["Linear","Log10"], horizontal=True, index=1,
                      help="Log10 is recommended: fine-tune between 1e-5 ~ 1e-1")
if scale_mode == "Linear":
    strength = st.slider("Quantization strength (delta scale)", min_value=1e-5, max_value=0.1, value=0.0015, step=1e-5,
                         help="Higher = more robust decoding, but more distortion. Lower = less distortion, but fragile.")
else:
    exp = st.slider("log10(strength)", min_value=-5.0, max_value=-1.0, value=-3.0, step=0.01,
                    help="Strength = 10^x. Example: x=-3 → 0.001, x=-2 → 0.01")
    strength = float(10**exp)
st.caption(f"Current strength ≈ {strength:.6f}")

if uploaded is not None:
    try:
        audio, sr = sf.read(uploaded)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float64)
        nfft = len(audio)  # 원본 오디오 길이 저장
        st.audio(uploaded, format="audio/wav")
        st.write(f"Sample rate: **{sr} Hz**, Duration: **{nfft/sr:.2f} s**, nfft: **{nfft} samples**")
    except Exception as e:
        st.error(f"Error reading audio: {e}")
        st.stop()

    if text:
        bits = text_to_bits(text)
        bits = add_length_prefix(bits)

        spec = rfft(audio)
        idx = pick_indices(sr, nfft, low_hz, high_hz, int(step_bins))
        capacity = len(idx)
        st.info(f"Available capacity: **{capacity} bits** / Required: **{len(bits)} bits**")

        if capacity < len(bits):
            st.error("Not enough capacity. Increase frequency range, decrease step size, or shorten the message.")
        else:
            col_delta1, col_delta2 = st.columns([1,1])
            with col_delta1:
                manual_delta_on = st.toggle("Manually set absolute delta (advanced)", value=False, help="Overrides strength-based estimate.")
            with col_delta2:
                delta_manual = st.number_input("Absolute delta", min_value=0.0, max_value=1.0, value=0.0, step=1e-6, format="%.9f")

            delta = float(delta_manual) if manual_delta_on and delta_manual > 0 else estimate_delta(spec, idx, strength)
            if delta <= 0:
                st.error("Invalid frequency band selection. Please adjust cutoffs.")
            else:
                apply_qim_encode(spec, idx[:len(bits)], bits, delta)
                st.write("spec[idx][:10] before:", spec[idx[:10]])
                stego = irfft(spec, n=nfft)
                mx = np.max(np.abs(stego))
                if mx > 0.999:
                    stego = stego / (mx + 1e-9) * 0.99

                buf = BytesIO()
                sf.write(buf, stego.astype(np.float32), sr, format="WAV", subtype="PCM_16")
                buf.seek(0)
                st.success("✅ Secret message successfully hidden in audio!")
                st.download_button("📥 Download stego audio (WAV)", data=buf, file_name="stego.wav", mime="audio/wav")

                with st.expander("Backup parameters", expanded=False):
                    st.code({
                        "low_hz": low_hz,
                        "high_hz": high_hz,
                        "step_bins": int(step_bins),
                        "strength": strength,
                        "estimated_delta": delta,
                        "sample_rate": sr,
                        "nfft": nfft,
                        "used_capacity_bits": int(len(bits))
                    }, language="json")
    else:
        st.warning("Please enter a text message to hide.")
else:
    st.info("Upload a WAV file from the sidebar.")
