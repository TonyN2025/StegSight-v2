# Pages/5_Audio_Detect.py
import streamlit as st
import pandas as pd
import tempfile

from features.audioFeatures import (
    make_waveform_fig, make_spectrogram_fig, make_scalogram_fig,
    extract_all_audio_features, suspicion_score
)

st.set_page_config(page_title="Audio Detect", page_icon="🧪")
st.title("🧪 Audio Steganalysis (Heuristic)")

uploaded = st.file_uploader("Upload WAV (16-bit PCM recommended)", type=["wav"])

col_cfg1, col_cfg2 = st.columns(2)
with col_cfg1:
    st.caption("This page shows waveform/spectrogram visuals and computes simple frequency-domain features.")
with col_cfg2:
    st.info("This is a **heuristic** detector. Replace with a trained ML model later for reliability.", icon="ℹ️")

if uploaded:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
        tmp_in.write(uploaded.read())
        tmp_in.flush()
        wav_path = tmp_in.name

    # Visuals
    st.subheader("Visualisations")
    try:
        st.pyplot(make_waveform_fig(wav_path), clear_figure=True)
        st.pyplot(make_spectrogram_fig(wav_path), clear_figure=True)
        fig_scale = make_scalogram_fig(wav_path)
        if fig_scale is not None:
            st.pyplot(fig_scale, clear_figure=True)
        else:
            st.caption("Scalogram skipped (PyWavelets not installed).")
    except Exception as e:
        st.warning(f"Could not render visuals: {e}")

    # Features
    st.subheader("Extracted Features")
    try:
        feats = extract_all_audio_features(wav_path)
        df = pd.DataFrame([feats])
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        feats = None

    # Heuristic suspicion score
    st.subheader("Suspicion Score (0–100)")
    if feats is not None:
        try:
            score, explain = suspicion_score(feats)
            st.metric("Suspicion", f"{score:.1f} / 100")
            with st.expander("What drove this score?"):
                st.write({
                    "High-frequency energy ratio": explain["hf_ratio"],
                    "Spectral flatness (mean)": explain["flatness_mean"],
                    "MFCC variability (mean std)": explain["mfcc_std_mean"],
                    "Normalised contributions": {
                        "hf": explain["norm_hf"],
                        "flatness": explain["norm_flat"],
                        "mfcc_var": explain["norm_mfcc_var"],
                    },
                    "Weights": explain["weights"],
                })
            st.caption("Tip: real detectors are trained on cover vs stego corpora; this score is a transparent heuristic placeholder.")
        except Exception as e:
            st.error(f"Scoring failed: {e}")
