import streamlit as st
import numpy as np
from PIL import Image, ImageFile
import io
import wave
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy

# ---------------------------------
# PIL: Strict image integrity
# ---------------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = False

# ---------------------------------
# Helper Functions
# ---------------------------------
def image_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

def bytes_to_image(b):
    return Image.open(io.BytesIO(b))

# ---- Header-Based Encoding ----
def encode_image_to_audio(image_bytes, audio_wave):
    audio_samples = np.frombuffer(audio_wave.readframes(audio_wave.getnframes()), dtype=np.int16)

    # Add header (32 bits for image byte length)
    length_header = format(len(image_bytes), '032b')
    bin_image = length_header + ''.join(format(byte, '08b') for byte in image_bytes)

    if len(bin_image) > len(audio_samples):
        raise ValueError(f"Audio too short! Need {len(bin_image)} samples, got {len(audio_samples)}")

    encoded_samples = audio_samples.copy()
    for i, bit in enumerate(bin_image):
        encoded_samples[i] = (encoded_samples[i] & ~1) | int(bit)

    return encoded_samples.tobytes()

# ---- Header-Based Decoding ----
def decode_image_from_audio(audio_wave):
    audio_samples = np.frombuffer(audio_wave.readframes(audio_wave.getnframes()), dtype=np.int16)
    bits = ''.join(str(sample & 1) for sample in audio_samples)

    # Read header (first 32 bits = image byte length)
    length_bits = bits[:32]
    img_length = int(length_bits, 2)

    # Extract exactly the needed number of bits
    img_bits = bits[32:32 + img_length * 8]
    if len(img_bits) < img_length * 8:
        raise ValueError("Truncated image data — possibly incomplete encoding or insufficient audio.")

    image_bytes = bytes(int(img_bits[i:i+8], 2) for i in range(0, len(img_bits), 8))
    return bytes_to_image(image_bytes)

# ---- Stego Detection ----
def detect_stego(audio_wave):
    audio_samples = np.frombuffer(audio_wave.readframes(audio_wave.getnframes()), dtype=np.int16)
    lsb_count = np.sum(audio_samples & 1)
    lsb_ratio = lsb_count / len(audio_samples)
    return 0.45 < lsb_ratio < 0.55

def save_encoded_wav(encoded_bytes, params):
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setparams(params)
        w.writeframes(encoded_bytes)
    buf.seek(0)
    return buf

# ---------------------------------
# Audio Visualisations
# ---------------------------------
def read_wav_samples_and_rate(file_like):
    file_like.seek(0)
    with wave.open(file_like, 'rb') as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sampwidth = w.getsampwidth()
        n = w.getnframes()
        if sampwidth != 2:
            raise ValueError(f"Unsupported sample width: {sampwidth*8} bits. Use 16-bit PCM WAV.")
        data = np.frombuffer(w.readframes(n), dtype=np.int16)
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1).astype(np.int16)
    return data, sr

def display_spectrogram(audio_samples, framerate, title="Spectrogram"):
    plt.figure(figsize=(10, 4))
    plt.specgram(audio_samples, NFFT=1024, Fs=framerate, noverlap=512)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    st.pyplot(plt)
    plt.close()

def display_fft(audio_samples, framerate, title="FFT Magnitude Spectrum"):
    n = len(audio_samples)
    if n == 0 or framerate <= 0:
        st.warning("No audio data to plot.")
        return
    fft_vals = np.fft.rfft(audio_samples)
    fft_freqs = np.fft.rfftfreq(n, 1.0 / framerate)
    mag = np.abs(fft_vals)

    plt.figure(figsize=(10, 4))
    plt.plot(fft_freqs, mag)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, framerate/2)
    st.pyplot(plt)
    plt.close()

# ---------------------------------
# Streamlit Interface
# ---------------------------------
st.title("🕵️‍♀️ Image-to-Audio Steganography Tool + Forensic Report")
option = st.selectbox("Choose operation:", ["Encode", "Decode", "Detect"])

# ---------------------------------
# Encode
# ---------------------------------
if option == "Encode":
    st.subheader("Encode Image into Audio")
    image_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
    audio_file = st.file_uploader("Upload WAV Audio", type=["wav"])

    if st.button("Encode") and image_file and audio_file:
        audio_file.seek(0)
        audio_wave = wave.open(audio_file, 'rb')
        num_samples = audio_wave.getnframes()
        img = Image.open(image_file)
        image_bytes = image_to_bytes(img)

        max_bytes = num_samples // 8 - 4  # account for 32-bit header
        if len(image_bytes) > max_bytes:
            scale_factor = (max_bytes / len(image_bytes))**0.5
            new_size = (int(img.width*scale_factor), int(img.height*scale_factor))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            image_bytes = image_to_bytes(img)
            st.warning(f"Image resized to {new_size} to fit audio length.")

        try:
            encoded_audio_bytes = encode_image_to_audio(image_bytes, audio_wave)
            audio_params = audio_wave.getparams()
            audio_wave.close()

            encoded_audio_file = save_encoded_wav(encoded_audio_bytes, audio_params)
            st.success("✅ Image encoded into audio successfully!")

            # ----- Audio Preview -----
            encoded_audio_file.seek(0)
            st.audio(encoded_audio_file, format="audio/wav")

            # ----- Visualisations -----
            samples_enc, sr_enc = read_wav_samples_and_rate(encoded_audio_file)
            st.markdown("**Visualisations**")
            display_spectrogram(samples_enc, sr_enc, title="Encoded Audio Spectrogram")
            display_fft(samples_enc, sr_enc, title="Encoded Audio FFT Spectrum")

            # ----- Download Button -----
            encoded_audio_file.seek(0)
            st.download_button("Download Encoded Audio", encoded_audio_file, "encoded_audio.wav", "audio/wav")

        except Exception as e:
            st.error(f"Error: {e}")

# ---------------------------------
# Decode
# ---------------------------------
elif option == "Decode":
    st.subheader("Decode Image from Audio")
    audio_file = st.file_uploader("Upload Encoded WAV Audio", type=["wav"])

    if st.button("Decode") and audio_file:
        audio_file.seek(0)
        try:
            with wave.open(audio_file,'rb') as audio_wave:
                img = decode_image_from_audio(audio_wave)

            if img:
                st.image(img, caption="Decoded Image")
                samples_dec, sr_dec = read_wav_samples_and_rate(audio_file)
                st.markdown("**Visualisations**")
                display_spectrogram(samples_dec, sr_dec, title="Decoded Audio Spectrogram")
                display_fft(samples_dec, sr_dec, title="Decoded Audio FFT Spectrum")
            else:
                st.warning("No hidden image found!")

        except Exception as e:
            st.error(f"Decoding failed: {e}")

# ---------------------------------
# Detect (with Forensic Report)
# ---------------------------------
elif option == "Detect":
    st.subheader("Detect Hidden Data in Audio")
    audio_file = st.file_uploader("Upload WAV Audio", type=["wav"])

    if st.button("Detect") and audio_file:
        audio_file.seek(0)
        with wave.open(audio_file, 'rb') as audio_wave:
            audio_samples = np.frombuffer(audio_wave.readframes(audio_wave.getnframes()), dtype=np.int16)
            sr = audio_wave.getframerate()

        # ---- LSB Analysis ----
        lsb = audio_samples & 1
        lsb_ratio = np.mean(lsb)
        lsb_entropy = - (lsb_ratio * np.log2(lsb_ratio + 1e-10) +
                         (1 - lsb_ratio) * np.log2(1 - lsb_ratio + 1e-10))
        lsb_score = abs(0.5 - lsb_ratio) * 200

        # ---- FFT Analysis ----
        fft_vals = np.fft.rfft(audio_samples)
        fft_mag = np.abs(fft_vals)
        high_freq_energy = np.mean(fft_mag[int(len(fft_mag)*0.8):])
        total_energy = np.mean(fft_mag)
        noise_ratio = high_freq_energy / (total_energy + 1e-10)
        fft_score = min(noise_ratio * 200, 100)

        # ---- Spectrogram Analysis ----
        spec, _, _, _ = plt.specgram(audio_samples, NFFT=1024, Fs=sr, noverlap=512)
        spec_entropy = shannon_entropy(spec)
        spec_score = min(spec_entropy / 10 * 100, 100)
        plt.close()

        # ---- Combine Forensic Scores ----
        suspiciousness = np.clip((lsb_score * 0.5 + fft_score * 0.3 + spec_score * 0.2), 0, 100)

        # ---- Display Visualisations ----
        st.markdown("### 🔍 Spectrogram & FFT Analysis")
        display_spectrogram(audio_samples, sr, title="Audio Spectrogram (Detection)")
        display_fft(audio_samples, sr, title="Audio FFT Spectrum (Detection)")

        # ---- Report Summary ----
        st.markdown("### 🧾 Forensic Report")
        col1, col2, col3 = st.columns(3)
        col1.metric("LSB Randomness", f"{lsb_entropy:.3f} bits")
        col2.metric("High-Frequency Noise Ratio", f"{noise_ratio:.3f}")
        col3.metric("Spectrogram Entropy", f"{spec_entropy:.2f}")

        st.markdown("---")
        if suspiciousness > 70:
            st.error(f"🚨 High likelihood of hidden data ({suspiciousness:.1f}%)")
        elif suspiciousness > 40:
            st.warning(f"⚠️ Moderate likelihood of hidden data ({suspiciousness:.1f}%)")
        else:
            st.success(f"✅ Low likelihood of hidden data ({suspiciousness:.1f}%)")

        # ---- Detailed text summary ----
        st.markdown(f"""
        **Analysis Summary:**
        - **LSB Randomness:** {lsb_entropy:.3f} bits (higher = more random)
        - **Noise Ratio:** {noise_ratio:.3f} (higher = more distortion)
        - **Spectrogram Entropy:** {spec_entropy:.2f} (higher = more complex)
        
        The suspiciousness score is derived from deviations in frequency balance,
        noise distribution, and randomness. A high score suggests possible
        steganographic alteration within the audio file.
        """)
