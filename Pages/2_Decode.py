import streamlit as st
from PIL import Image
from pathlib import Path
import os
import io

st.set_page_config(page_title="Uncover", page_icon="🔎")

st.title("🔎 Uncover")

uploaded_files = st.file_uploader("Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

def decode_lsb(image: Image.Image) -> bytes:
    # Decode hidden message using LSB steganography (1 bit per pixel in RGB channels)
    pixels = image.convert("RGB").getdata()
    bits = []
    for pixel in pixels:
        for color in pixel:
            bits.append(color & 1)
    # Group bits into bytes
    bytes_list = []
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i+8]
        if len(byte_bits) < 8:
            break
        byte = 0
        for bit in byte_bits:
            byte = (byte << 1) | bit
        bytes_list.append(byte)
    data = bytes(bytes_list)
    # Try to find a null terminator or a reasonable cutoff
    if b'\x00' in data:
        data = data[:data.index(b'\x00')]
    return data

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = uploads_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            img = Image.open(file_path)
        except Exception as e:
            st.error(f"Error opening image: {e}")
            continue

        with st.expander(f"Preview of {uploaded_file.name}", expanded=True):
            st.image(img, use_container_width=True)

        try:
            extracted_bytes = decode_lsb(img)
            if not extracted_bytes:
                st.info("No hidden message detected.")
                continue
            # Heuristic: if extracted_bytes is mostly printable ASCII, show as text else hex
            try:
                extracted_text = extracted_bytes.decode("utf-8")
                display_text = extracted_text[:500]
                st.markdown(
                    f"Extracted hidden message of <span style='color:#1E90FF;'>{uploaded_file.name}</span> (first 500 characters):",
                    unsafe_allow_html=True
                )
                st.text_area("", display_text, height=200)
                payload_bytes = extracted_bytes
            except UnicodeDecodeError:
                display_hex = extracted_bytes[:64].hex()
                st.markdown(
                    f"Extracted hidden message of <span style='color:#1E90FF;'>{uploaded_file.name}</span> (first 64 bytes in hex):",
                    unsafe_allow_html=True
                )
                st.text_area("", display_hex, height=200)
                payload_bytes = extracted_bytes

        except Exception as e:
            st.error(f"Error decoding hidden message: {e}")
