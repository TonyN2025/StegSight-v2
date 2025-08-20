import streamlit as st
from PIL import Image

# TODO: add ML
# For now, this page can handle image uploads
st.title("🔐 Encode")
st.write("Hide a secret message inside an image.")

uploaded_file = st.file_uploader("Upload a cover image", type=["png", "jpg", "jpeg"])
message = st.text_area("Enter your secret message:")

if uploaded_file and message:
    st.success("Encoding would happen here...")
