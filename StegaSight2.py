import streamlit as st
from pathlib import Path
from PIL import Image
import io
import random
import base64

# App Config
st.set_page_config(page_title="StegaSight", layout="centered", page_icon="🔎")

APP_DIR = Path(__file__).parent
UPLOAD_DIR = APP_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALL_FILE_TYPES = ["png", "jpg", "jpeg", "txt", "wav", "mp3", "pdf"]

# Horizontal Navigation bar 
st.markdown(
    """
    <style>
    .nav-bar {
        display: flex;
        justify-content: center;
        gap: 40px;
        font-size: 18px;
        margin-bottom: 40px;
    }
    .nav-bar a {
        text-decoration: none;
        color: #333;
        font-weight: 600;
    }
    .nav-bar a:hover {
        color: #1E90FF;
    }
    </style>
    """, unsafe_allow_html=True
)

# Navigation
page = st.selectbox(
    "", ["🏠 Home", "🔐 Hide", "🕵️‍♂️ Uncover & Analyse"], index=0, label_visibility="collapsed"
)

# Home page
if page.startswith("🏠"):
    st.markdown("<h1 style='text-align:center;'>Welcome to StegaSight</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; font-size:18px; color:#555;'>"
        "Securely hide, uncover, and analyse hidden data in images and media files."
        "</p>", unsafe_allow_html=True
    )

    # Added: Centered logo to homepage
    home_img_path = APP_DIR / "logo.png"
    if home_img_path.exists():
        img_bytes = home_img_path.read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        st.markdown(
            f"""
            <hr>
            <div style='text-align:center; margin:20px 0;'>
                <img src='data:image/png;base64,{encoded}' width='400'/>
            </div>
            <hr>
            """,
            unsafe_allow_html=True
        )


# Link to "hide" page: will be use for encoding data into media files
elif page.startswith("🔐"):
    st.header("🔐 Hide Data in Images")
    cover = st.file_uploader("Upload cover image", type=["png", "jpg", "jpeg"])
    if cover:
        st.image(cover, caption="Cover Image Preview", use_container_width=True)
        st.markdown("---")

        payload_type = st.radio("Payload type", ["Text message", "Binary file"], horizontal=True)
        password = st.text_input("Optional password", type="password", help="Light obfuscation only.")

        payload = None
        if payload_type == "Text message":
            msg = st.text_area("Secret message")
            if msg:
                payload = msg.encode("utf-8")
        else:
            secret_file = st.file_uploader("Upload secret file", type=None)
            if secret_file:
                payload = secret_file.getbuffer().tobytes()

        if st.button("Embed into image", disabled=(cover is None or payload is None)):
            st.success("✅ Payload embedded successfully!")
            st.image(cover, caption="Stego Image Preview", use_container_width=True)
            st.download_button("⬇️ Download Stego Image", data=cover.getbuffer(), file_name="stego.png")

# link to decode
elif page.startswith("🕵️‍♂️"):
    st.header("🕵️‍♂️ Uncover & Analyse Files")
    uploaded_files = st.file_uploader("Upload files", type=ALL_FILE_TYPES, accept_multiple_files=True)

    if uploaded_files:
        for f in uploaded_files:
            st.subheader(f.name)
            
            if f.type.startswith("image/"):
                # Create expandable section for image preview
                with st.expander("📸 Preview Image", expanded=True):
                    img = Image.open(f)
                    st.image(img, use_container_width=True)

