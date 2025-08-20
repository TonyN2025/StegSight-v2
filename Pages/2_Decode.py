import streamlit as st
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Your existing imports
from features import extract_basic_stats, extract_spectrogram_features, extract_advanced_steganalysis_features, assess_steganalysis_threat
from StegaSight2 import decode_lsb, detect_threat, UPLOAD_DIR, DECODE_DIR

st.set_page_config(page_title="Decode & Analyse", layout="wide")

st.title("🕵️ Decode & Analyse")
st.markdown("Upload images or audio, we'll try to decode hidden payloads and analyse them.")
st.header("🕵️‍♂️ Uncover & Analyse Files")
uploaded_files = st.file_uploader("Upload files", type=ALL_FILE_TYPES, accept_multiple_files=True)

# Initialize session state
if 'decoded_payload' not in st.session_state:
    st.session_state.decoded_payload = None
if 'decoded_filename' not in st.session_state:
    st.session_state.decoded_filename = None

def create_steganalysis_visualizations(img: Image.Image, features: dict):
    """Create visualizations for steganalysis results"""
    gray_array = np.array(img.convert('L'))
    rgb_array = np.array(img)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Grayscale histogram
    axes[0, 0].hist(gray_array.flatten(), bins=256, alpha=0.7, color='blue')
    axes[0, 0].set_title('Grayscale Histogram')
    axes[0, 0].set_xlabel('Pixel Intensity')
    axes[0, 0].set_ylabel('Frequency')
    
    # LSB plane
    lsb_plane = gray_array & 1
    axes[0, 1].imshow(lsb_plane, cmap='gray')
    axes[0, 1].set_title('LSB Plane')
    axes[0, 1].axis('off')
    
    # FFT spectrum
    fft_transform = np.fft.fft2(gray_array)
    fft_magnitude = np.log(np.abs(np.fft.fftshift(fft_transform)) + 1)
    axes[0, 2].imshow(fft_magnitude, cmap='hot')
    axes[0, 2].set_title('Frequency Spectrum')
    axes[0, 2].axis('off')
    
    # Color channel histograms
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        channel_data = rgb_array[:, :, i].flatten()
        axes[1, i].hist(channel_data, bins=64, color=color, alpha=0.7)
        axes[1, i].set_title(f'{color.upper()} Channel')
        axes[1, i].set_xlabel('Pixel Intensity')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

uploaded_files = st.file_uploader(
    "📥 Upload your files",
    type=["png", "jpg", "jpeg", "wav", "mp3"],
    accept_multiple_files=True,
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = UPLOAD_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ {uploaded_file.name} uploaded.")

        file_type = uploaded_file.type

        with st.expander(f"📖 Preview & Decode: {uploaded_file.name}", expanded=True):
            if file_type.startswith("image/"):
                image = Image.open(file_path)

                with st.expander("🖼 Image Preview", expanded=True):
                    st.image(image, caption="Uploaded image", use_container_width=True)
                
                # Enhanced steganalysis
                st.header("🔍 Advanced Steganalysis")
                
                # Extract advanced features
                advanced_features = extract_advanced_steganalysis_features(image)
                
                # Display features in a nice format
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Feature Summary")
                    features_df = pd.DataFrame([advanced_features])
                    st.dataframe(features_df.T.style.background_gradient(cmap='viridis'))
                
                with col2:
                    st.subheader("🛡️ Threat Assessment")
                    threat_level, indicators = assess_steganalysis_threat(advanced_features)
                    
                    threat_colors = {"safe": "green", "suspicious": "orange", "dangerous": "red"}
                    st.markdown(f"<h3 style='color: {threat_colors[threat_level]}'>Threat Level: {threat_level.upper()}</h3>", unsafe_allow_html=True)
                    
                    if indicators:
                        st.write("**Detection Indicators:**")
                        for indicator in indicators:
                            st.write(f"• {indicator}")
                    else:
                        st.write("No suspicious indicators detected.")
                
                # Visualizations
                st.subheader("📈 Visual Analysis")
                create_steganalysis_visualizations(image, advanced_features)
                
                # Existing decode form
                with st.form(key=f"decode_form_{uploaded_file.name}"):
                    dec_pwd = st.text_input("Password (if used)", type="password")
                    submitted = st.form_submit_button("Try decode")
                    if submitted:
                        try:
                            data = decode_lsb(image, password=dec_pwd or None)
                            st.success("Decoded payload found!")
                            st.session_state.decoded_payload = data
                            st.session_state.decoded_filename = f"extracted_{Path(uploaded_file.name).stem}.bin"
                            
                            try:
                                snippet = data.decode("utf-8")[:500]
                                st.code(snippet)
                            except Exception:
                                st.info("Payload appears binary:")
                                st.code(data[:64].hex(" "))
                            
                            out_path = DECODE_DIR / st.session_state.decoded_filename
                            with open(out_path, "wb") as w:
                                w.write(data)
                        except Exception as e:
                            st.warning(f"No decodable payload or wrong password. ({e})")
                            st.session_state.decoded_payload = None
                            st.session_state.decoded_filename = None
                
                # Basic features (original functionality)
                st.subheader("📊 Basic Image Features")
                features_basic = extract_basic_stats(image)
                st.write(features_basic)

            elif file_type.startswith("audio/"):
                st.audio(str(file_path))
                st.subheader("📊 Extracted Audio Features")
                try:
                    features = extract_spectrogram_features(str(file_path))
                    st.write(features.tolist())
                except Exception as e:
                    st.error(f"Audio feature extraction failed: {e}")

        # Original threat detection
        original_threat_level, extracted_content = detect_threat(file_path, file_type)
        badge = {"safe": "💚 Safe", "danger": "💔 Dangerous"}.get(original_threat_level, original_threat_level)
        st.write(f"**Original Verdict:** {badge}")

        if extracted_content:
            st.markdown("**Tool outputs:**")
            for tool_name, content in extracted_content.items():
                with st.expander(f"🔍 {tool_name}"):
                    st.text(content or "(no output)")