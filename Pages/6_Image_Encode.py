import streamlit as st
from PIL import Image
import numpy as np
import io

st.markdown(
    "⚠️ **Note:** Please keep images around 200x200 px. Higher resolution images require longer processing times due to Python's slower handling. "
    "Use the 'Hidden bits' slider below to showcase the transformation for educational purposes."
)

st.title("🔐 Image Encoder & Decoder")

col1, col2 = st.columns(2)

with col1:
    cover_image_file = st.file_uploader("Upload Cover Image", type=["png", "jpg", "jpeg"], key="cover")
    cover_img = None
    if cover_image_file is not None:
        try:
            cover_img = Image.open(cover_image_file).convert("RGB")
            st.image(cover_img, caption="Cover Image", use_container_width=True)
            if cover_img.width < 200 or cover_img.height < 200:
                st.warning("Cover image is small; hidden image may not be visible until larger sizes.")
        except Exception as e:
            st.error(f"Error loading cover image: {e}")

with col2:
    hidden_image_file = st.file_uploader("Upload Image to be Hidden", type=["png", "jpg", "jpeg"], key="hidden")
    hidden_img = None
    if hidden_image_file is not None:
        try:
            hidden_img = Image.open(hidden_image_file).convert("RGB")
            st.image(hidden_img, caption="Image to be Hidden", use_container_width=True)
        except Exception as e:
            st.error(f"Error loading hidden image: {e}")

# Default hidden_bits if images not uploaded
hidden_bits_default = 2

# Slider visible in all cases
hidden_bits = st.slider(
    "Hidden bits",
    min_value=1,
    max_value=20,
    value=hidden_bits_default,
    help="Number of least significant bits to encode/decode. Adjust to see visual effects."
)

def bits_to_int(bits):
    return int(bits, 2)

def int_to_bits(value, length):
    return bin(value)[2:].zfill(length)

def encode_images(cover_img, hidden_img, n_bits):
    cover_img = cover_img.convert('RGB')
    hidden_img = hidden_img.convert('RGB')

    cover_np = np.array(cover_img)
    hidden_np = np.array(hidden_img)

    h_cover, w_cover, _ = cover_np.shape
    h_hidden, w_hidden, _ = hidden_np.shape

    # Resize hidden image if larger than cover image
    if h_hidden > h_cover or w_hidden > w_cover:
        new_width = min(w_hidden, w_cover)
        new_height = min(h_hidden, h_cover)
        hidden_img = hidden_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        hidden_np = np.array(hidden_img)
        h_hidden, w_hidden = hidden_np.shape[:2]
        st.warning(f"Hidden image resized to {new_width}x{new_height} to fit into cover image.")

    # Embed secret image MSBs into cover image LSBs
    bit_mask_cover = 0xFF ^ ((1 << n_bits) - 1)  # clear n LSBs of cover
    shift_amount = 8 - n_bits  # number of MSBs to use from secret

    for i in range(min(h_hidden, h_cover)):
        for j in range(min(w_hidden, w_cover)):
            for c in range(3):  # RGB channels
                cover_val = cover_np[i, j, c]
                secret_val = hidden_np[i, j, c]
                # Extract n MSBs from secret
                secret_msb = (secret_val >> shift_amount) & ((1 << n_bits) - 1)
                # Clear n LSBs of cover and insert secret MSBs
                cover_np[i, j, c] = (cover_val & bit_mask_cover) | secret_msb

    stego_img = Image.fromarray(cover_np.astype(np.uint8))

    return stego_img

def decode_bits(stego_img, n_bits):
    stego_img = stego_img.convert('RGB')
    np_stego = np.array(stego_img)
    h, w, c = np_stego.shape

    secret_np = np.zeros((h, w, 3), dtype=np.uint8)
    shift_amount = 8 - n_bits
    mask = (1 << n_bits) - 1

    for i in range(h):
        for j in range(w):
            for c in range(3):
                # Extract n LSBs and shift left to approximate original MSBs
                val = np_stego[i, j, c] & mask
                secret_np[i, j, c] = val << shift_amount

    secret_img = Image.fromarray(secret_np)
    return secret_img

stego_img = None
error_message = None

if cover_img is not None and hidden_img is not None:
    try:
        stego_img = encode_images(cover_img, hidden_img, hidden_bits)
    except Exception as e:
        error_message = str(e)
        stego_img = None

if stego_img is not None:
    st.subheader("Stego Image (Cover + Hidden)")
    st.image(stego_img, use_container_width=True)

    buf = io.BytesIO()
    stego_img.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="Download Stego Image",
        data=byte_im,
        file_name="stego_image.png",
        mime="image/png"
    )

if error_message is not None:
    st.error(f"Error: {error_message}")
