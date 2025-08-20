import streamlit as st
from pathlib import Path
import base64

APP_DIR = Path(__file__).parent
UPLOAD_DIR = APP_DIR / "uploads"
DECODE_DIR = APP_DIR / "decoded"
UPLOAD_DIR.mkdir(exist_ok=True)
DECODE_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":
    # App Config
    st.set_page_config(page_title="StegaSight", layout="centered", page_icon="🔎")


    # Home page
    st.markdown(
        "<h1 style='text-align:center; font-size:56px; font-weight:bold; font-family:sans-serif; margin-bottom:0;'>StegaSight</h1>"
        "<p style='text-align:center; font-size:18px; color:#999; margin-top:4px; font-family:sans-serif;'>hide, uncover, and analyse</p>",
        unsafe_allow_html=True
    )

    # Centered logo to homepage
    home_img_path = APP_DIR / "logo.png"
    if home_img_path.exists():
        img_bytes = home_img_path.read_bytes()
        encoded = base64.b64encode(img_bytes).decode()
        st.markdown(
            f"""
            <hr>
            <div style='text-align:center; margin:20px 0;'>
                <img src='data:image/png;base64,{encoded}' width='250'/>
            </div>
            <hr>
            """,
            unsafe_allow_html=True
        )
