import streamlit as st
from stegano import lsb
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="Encode", page_icon="🔒")

st.title("🔐 Hide Data")

uploaded_cover = st.file_uploader("Upload a cover image", type=["png", "jpg", "jpeg"])

secret_text = st.text_area("Enter a secret message to hide")

if uploaded_cover and secret_text:
    try:
        cover_bytes = uploaded_cover.read()
        image = Image.open(BytesIO(cover_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        with st.expander("Cover Image", expanded=True):
            st.image(image, use_container_width=True)

        encoded_image = lsb.hide(image, secret_text)

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

        st.success("Secret message hidden successfully in the cover image!")
    except Exception as e:
        st.error(f"Error encoding secret message into cover image: {e}")
elif uploaded_cover and not secret_text:
    st.warning("Please enter a secret message to hide.")
