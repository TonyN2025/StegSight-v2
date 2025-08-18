import streamlit as st

st.title("🔎 Decode")
st.write("Extract hidden messages from an image.")

uploaded_file = st.file_uploader("Upload an image with hidden data", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.success("Decoding would happen here...")
    # Still need to add ML decoding logic

st.title("📊 Analyse")
st.write("Run steganalysis on an image to detect hidden data.")

uploaded_file = st.file_uploader("Upload an image for analysis", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.success("Analysis would happen here...")
    # Still need to add ML analysis logic
