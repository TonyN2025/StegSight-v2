import streamlit as st
import tempfile
from features.audio_lsb import embed_lsb_wav

st.set_page_config(page_title="Audio Encode", page_icon="🎵")
st.title("🎵 Hide Text in Audio (LSB)")

uploaded = st.file_uploader("Upload cover audio (WAV, 16-bit PCM)", type=["wav"])
secret = st.text_area("Secret message")
col1, col2, col3 = st.columns([2,1,1])

with col1:
    key = st.text_input("Key (for scattering)", value="audio")
with col2:
    channel = st.selectbox("Channel", options=[0, 1], index=0, help="If stereo, choose which channel to use.")
with col3:
    max_ratio = st.number_input("Max ratio", min_value=0.0, max_value=1.0, value=0.5,
                                help="Cap payload bits as fraction of capacity (lower = harder to detect).")

if uploaded and secret:
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            tmp_in.write(uploaded.read()); tmp_in.flush()
            embed_lsb_wav(tmp_in.name, tmp_out.name, secret, key=key, channel=channel, max_ratio=max_ratio)
            st.audio(tmp_out.name, format="audio/wav")
            with open(tmp_out.name, "rb") as f:
                st.download_button("Download stego WAV", f, file_name="stego.wav", mime="audio/wav")
        st.success("Message embedded successfully.")
    except Exception as e:
        st.error(f"Encoding failed: {e}")
elif uploaded and not secret:
    st.info("Enter a secret message to embed.")
