import streamlit as st      #web-interface
from PIL import Image       #for importing images
from pathlib import Path    #for path operations
from stegano import lsb
import base64
import io

st.set_page_config(page_title="Uncovering", page_icon="🔎")

st.title("🔎 Find what's hidden")

uploaded_files = st.file_uploader("Upload your file", type=["png", "jpg", "jpeg", "mp4", "wav", "mp3"], accept_multiple_files=True)

uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

def decode_lsb(image: Image.Image):
    # Use stegano.lsb.reveal() to decode hidden message
    revealed = lsb.reveal(image)
    if revealed is None:
        return None
    if revealed.startswith("MSG:"):
        # Return the message text after 'MSG:'
        return revealed.encode('utf-8')
    elif revealed.startswith("FILE:"):
        # Decode base64 content after 'FILE:' prefix
        b64_content = revealed[5:]
        try:
            return base64.b64decode(b64_content)
        except Exception:
            return None
    else:
        # Treat as UTF-8 text
        return revealed.encode('utf-8')

def decode_media(file_bytes: bytes):
    # Search for marker ---SECRET FILE START---
    marker = b"---SECRET FILE START---"
    idx = file_bytes.find(marker)
    if idx == -1:
        return None
    b64_content = file_bytes[idx+len(marker):].strip()
    try:
        return base64.b64decode(b64_content)
    except Exception:
        return None

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = uploads_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        file_ext = uploaded_file.name.lower().split('.')[-1]

        if file_ext in ['png', 'jpg', 'jpeg']:
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
            except Exception as e:
                st.error(f"Error decoding hidden message: {e}")
                continue

        elif file_ext in ['mp4', 'wav', 'mp3']:
            try:
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
                extracted_bytes = decode_media(file_bytes)
                if not extracted_bytes:
                    st.info("No hidden message detected.")
                    continue
            except Exception as e:
                st.error(f"Error decoding hidden message: {e}")
                continue
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
            continue

        # Determine if extracted_bytes is text or file
        try:
            extracted_text = extracted_bytes.decode("utf-8")
            if extracted_text.startswith("MSG:"):
                # Text message
                message = extracted_text[4:]
                display_text = message[:500]
                st.markdown(
                    f"Extracted hidden message of <span style='color:#1E90FF;'>{uploaded_file.name}</span> (first 500 characters):",
                    unsafe_allow_html=True
                )
                st.text_area("", display_text, height=200)
            elif extracted_text.startswith("FILE:"):
                # File content in base64 after 'FILE:' prefix
                b64_content = extracted_text[5:]
                try:
                    file_content = base64.b64decode(b64_content)
                except Exception:
                    st.error("Failed to decode hidden file content.")
                    continue

                # Try to detect if file_content is an image
                try:
                    recovered_img = Image.open(io.BytesIO(file_content))
                    ext = recovered_img.format.lower() if recovered_img.format else "png"
                    st.markdown(
                        f"Extracted hidden image file of <span style='color:#1E90FF;'>{uploaded_file.name}</span>:",
                        unsafe_allow_html=True
                    )
                    st.image(recovered_img, use_container_width=True)
                    st.download_button(
                        label="Download extracted image",
                        data=file_content,
                        file_name=f"extracted_file.{ext}",
                        mime=f"image/{ext}"
                    )
                except Exception:
                    display_hex = file_content[:64].hex()
                    st.markdown(
                        f"Extracted hidden file of <span style='color:#1E90FF;'>{uploaded_file.name}</span> (first 64 bytes in hex):",
                        unsafe_allow_html=True
                    )
                    st.text_area("", display_hex, height=200)
                    st.download_button(
                        label="Download extracted file",
                        data=file_content,
                        file_name="extracted_file",
                        mime="application/octet-stream"
                    )
            else:
                # Treat as text message
                display_text = extracted_text[:500]
                st.markdown(
                    f"Extracted hidden message of <span style='color:#1E90FF;'>{uploaded_file.name}</span> (first 500 characters):",
                    unsafe_allow_html=True
                )
                st.text_area("", display_text, height=200)
        except UnicodeDecodeError:
            # Binary file
            display_hex = extracted_bytes[:64].hex()
            st.markdown(
                f"Extracted hidden file of <span style='color:#1E90FF;'>{uploaded_file.name}</span> (first 64 bytes in hex):",
                unsafe_allow_html=True
            )
            st.text_area("", display_hex, height=200)
            st.download_button(
                label="Download extracted file",
                data=extracted_bytes,
                file_name="extracted_file",
                mime="application/octet-stream"
            )
