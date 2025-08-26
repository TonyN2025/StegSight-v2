import streamlit as st
import tempfile
from features.audio_lsb import extract_lsb_wav

st.set_page_config(page_title="Audio Decode", page_icon="🔎")
st.title("🔎 Extract Text from Audio (LSB)")

uploaded = st.file_uploader("Upload stego audio (WAV, 16-bit PCM)", type=["wav"])
key = st.text_input("Key", value="audio")
channel = st.selectbox("Channel", options=[0, 1], index=0)

if uploaded:
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_in:
            tmp_in.write(uploaded.read()); tmp_in.flush()
            text = extract_lsb_wav(tmp_in.name, key=key, channel=channel)
        if text:
            st.success("Recovered message:")
            st.text_area("", text, height=200)
        else:
            st.warning("No message recovered. Check key/channel, or file may not be LSB-embedded.")
    except Exception as e:
        st.error(f"Decoding failed: {e}")
