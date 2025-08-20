import streamlit as st
from stegano import lsb
from PIL import Image
import base64
from io import BytesIO
import numpy as np

st.set_page_config(page_title="Encode", page_icon="🔒")

st.title("🔐 Hide Data")

uploaded_cover = st.file_uploader("Upload a cover file", type=["png", "jpg", "jpeg", "mp4", "wav", "mp3"])

secret_text = st.text_area("Enter a secret message to hide")

secret_file = st.file_uploader("Upload a secret file (can be any file type, including images)", type=None)

num_bits = None
if secret_file is not None:
    try:
        secret_file.seek(0)
        secret_file_bytes = secret_file.read()
        secret_file.seek(0)
        secret_file_image = Image.open(secret_file)
        if secret_file_image.mode != 'RGB':
            secret_file_image = secret_file_image.convert('RGB')
        num_bits = st.slider("Select number of bits to hide (1-4)", min_value=1, max_value=4, value=2)
    except Exception:
        secret_file_image = None
else:
    secret_file_image = None

def merge_images_n_lsb(cover_img: Image.Image, secret_img: Image.Image, n_bits: int) -> Image.Image:
    cover = np.array(cover_img, dtype=np.uint8)
    secret = np.array(secret_img, dtype=np.uint8)

    shift = 8 - n_bits
    # Take the n most significant bits of the secret image
    secret_msb = (secret >> shift) & ((1 << n_bits) - 1)
    # Clear the n least significant bits of the cover image
    cover_cleared = cover & (0xFF << n_bits)
    # Combine cover and secret
    combined = cover_cleared | secret_msb
    combined = np.clip(combined, 0, 255).astype(np.uint8)

    return Image.fromarray(combined)

if uploaded_cover:
    cover_bytes = uploaded_cover.read()
    cover_ext = uploaded_cover.name.split('.')[-1].lower()

    secret_data = None
    secret_prefix = None

    if secret_file is not None and secret_file_image is not None:
        try:
            cover_image = Image.open(BytesIO(cover_bytes))
            if cover_image.mode != 'RGB':
                cover_image = cover_image.convert('RGB')

            secret_img_pil = secret_file_image
            # Resize secret image to match cover image size
            if secret_img_pil.size != cover_image.size:
                secret_img_pil = secret_img_pil.resize(cover_image.size)

            encoded_image = merge_images_n_lsb(cover_image, secret_img_pil, num_bits)

            cols = st.columns(3)
            with cols[0]:
                with st.expander("Cover Image", expanded=True):
                    st.image(cover_image, use_container_width=True)
            with cols[1]:
                with st.expander("Secret Image", expanded=True):
                    st.image(secret_img_pil, use_container_width=True)
            with cols[2]:
                with st.expander("Encoded Image", expanded=True):
                    st.image(encoded_image, use_container_width=True)
                    buf = BytesIO()
                    encoded_image.save(buf, format='PNG')
                    buf.seek(0)
                    st.download_button(
                        label="Download encoded image",
                        data=buf,
                        file_name="encoded_image.png",
                        mime="image/png"
                    )

            st.success("Secret image hidden successfully in the cover image!")
        except Exception as e:
            st.error(f"Error encoding secret image into cover image: {e}")

    else:
        if secret_file is not None and secret_file_image is None:
            secret_bytes = secret_file.read()
            secret_b64 = base64.b64encode(secret_bytes).decode()
            secret_data = f"FILE:{secret_b64}"
            secret_prefix = "FILE:"
        elif secret_text:
            secret_data = f"MSG:{secret_text}"
            secret_prefix = "MSG:"

        if secret_data is None:
            st.warning("Please enter a secret message or upload a secret file to hide.")
        else:
            if cover_ext in ['png', 'jpg', 'jpeg']:
                image = Image.open(BytesIO(cover_bytes))
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                with st.expander("Cover Image", expanded=True):
                    st.image(image, use_container_width=True)

                try:
                    encoded_image = lsb.hide(image, secret_data)
                    buf = BytesIO()
                    encoded_image.save(buf, format='PNG')
                    buf.seek(0)
                    st.success("Data hidden successfully in the image!")
                    st.download_button(
                        label="Download encoded image",
                        data=buf,
                        file_name="encoded_image.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"Error encoding data into image: {e}")

            elif cover_ext in ['mp4', 'wav', 'mp3']:
                marker = b"\n---SECRET FILE START---\n"
                try:
                    secret_bytes_to_append = secret_data.encode()
                    secret_b64_bytes = secret_bytes_to_append if secret_prefix == "MSG:" else secret_data.encode()
                    
                    # Actually secret_data is string, so encode it
                    secret_b64_bytes = secret_data.encode()
                    new_file_bytes = cover_bytes + marker + secret_b64_bytes

                    st.success("Data hidden successfully in the file!")
                    st.download_button(
                        label="Download encoded file",
                        data=new_file_bytes,
                        file_name=f"encoded_{uploaded_cover.name}",
                        mime=uploaded_cover.type if uploaded_cover.type else "application/octet-stream"
                    )
                except Exception as e:
                    st.error(f"Error encoding data into file: {e}")

            else:
                st.error("Unsupported cover file type.")
