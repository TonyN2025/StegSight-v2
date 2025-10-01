import streamlit as st
import numpy as np
import soundfile as sf
from numpy.fft import rfft
import string

st.set_page_config(page_title="Audio Steganalysis (Fourier-QIM)", page_icon="🕵️", layout="centered")
st.title("🕵️ Detect Hidden Text in Audio (Fourier/QIM)")

st.markdown("""
This page attempts to **detect and extract hidden text from audio** using Quantization Index Modulation (QIM) in the frequency domain, **without prior knowledge of encoding parameters**.
- It automatically tries various parameter combinations and ranks extracted messages by plausibility.
- Results may contain false positives if the audio is not steganographic.
""")

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
        return ""

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

def plausibility_score(txt):
    if not txt: return 0
    printable = sum(c in string.printable for c in txt)
    ratio = printable / max(1, len(txt))
    bonus = 0
    if len(txt) > 10: bonus += 0.2
    if ' ' in txt: bonus += 0.2
    if "<decode error>" in txt: return 0
    return ratio + bonus

uploaded = st.file_uploader("Upload audio file to analyze (WAV)", type=["wav"])

if uploaded is not None:
    try:
        audio, sr = sf.read(uploaded)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float64)
        st.audio(uploaded, format="audio/wav")
        st.write(f"Sample rate: **{sr} Hz**, Duration: **{len(audio)/sr:.2f} s**")
    except Exception as e:
        st.error(f"Error reading audio: {e}")
        st.stop()

    st.info("Scanning for hidden messages... This may take a few seconds.")

    low_hz_list = [4000, 8000, 12000]
    high_hz_list = [14000, 18000, min(sr//2-1, 22000)]
    step_bins_list = [2, 4, 8]
    strength_list = [0.0005, 0.0015, 0.005, 0.01]
    delta_scale_mode = ["Linear", "Log10"]

    results = []

    spec = rfft(audio)
    nfft = len(audio)
    for low_hz in low_hz_list:
        for high_hz in high_hz_list:
            for step_bins in step_bins_list:
                idx = pick_indices(sr, nfft, low_hz, high_hz, step_bins)
                if len(idx) == 0: continue
                for strength in strength_list:
                    mags = np.abs(spec[idx])
                    base = np.percentile(mags, 70)
                    delta = max(base * strength, 1e-9)
                    bits = qim_decode(spec, idx, delta, max_bits=len(idx))
                    txt = bits_to_text(bits)
                    score = plausibility_score(txt)
                    if score > 0.4 and len(txt) > 0:
                        results.append({
                            "score": score,
                            "text": txt,
                            "low_hz": low_hz,
                            "high_hz": high_hz,
                            "step_bins": step_bins,
                            "strength": strength,
                            "estimated_delta": delta,
                            "length": len(txt)
                        })

    results = sorted(results, key=lambda x: -x["score"])
    if results:
        st.success(f"Found {len(results)} likely hidden message candidates!")
        for i, r in enumerate(results[:5]):
            st.markdown(f"**Candidate #{i+1}** (score={r['score']:.2f}, length={r['length']})")
            st.text_area("Decoded message", value=r["text"], height=140)
            st.code({k: r[k] for k in ["low_hz", "high_hz", "step_bins", "strength", "estimated_delta"]}, language="json")
    else:
        st.warning("No plausible hidden messages found. The audio may not contain steganography or it used different parameters.")
else:
    st.info("Upload a WAV file from the sidebar.")
