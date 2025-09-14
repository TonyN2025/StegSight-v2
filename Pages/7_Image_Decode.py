import streamlit as st
from PIL import Image
import numpy as np
from io import BytesIO

def decode_image(stego_img: Image.Image, n_lsb: int) -> Image.Image:
    """Extract hidden image from stego image using n LSBs per color channel."""
    stego_img = stego_img.convert('RGB')
    np_stego = np.array(stego_img).astype(np.uint8)
    h, w, _ = np_stego.shape

    # Extract n_lsb bits from each color channel and shift left to reconstruct approximate pixel value
    mask = (1 << n_lsb) - 1
    extracted_bits = np_stego & mask  # Extract n_lsb bits
    shift_amount = 8 - n_lsb
    extracted_pixels = extracted_bits << shift_amount

    # Reconstruct hidden image
    hidden_img = Image.fromarray(extracted_pixels.astype(np.uint8))
    return hidden_img

st.title("🔍 Image in Image Decoder 🕵️‍♂️")
st.write("Extract a hidden image from a stego image")

uploaded_stego = st.file_uploader("Upload a stego image", type=["png", "jpg", "jpeg"])

if uploaded_stego:
    try:
        stego_image = Image.open(uploaded_stego)
        st.image(stego_image, caption="Stego Image", use_container_width=True)

        n_lsb = st.slider("Hidden bits", min_value=1, max_value=8, value=2)

        decoded_img = decode_image(stego_image, n_lsb)
        st.image(decoded_img, caption="Decoded Hidden Image", use_container_width=True)

        buf = BytesIO()
        decoded_img.save(buf, format="PNG")
        buf.seek(0)

        st.download_button(
            label="⬇️ Download Decoded Hidden Image",
            data=buf,
            file_name="decoded_hidden_image.png",
            mime="image/png"
        )
    except Exception as e:
        st.error(f"Error decoding hidden image: {e}")
else:
    st.info("Please upload a stego image to decode.")