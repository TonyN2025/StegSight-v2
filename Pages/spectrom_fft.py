import streamlit as st
import numpy as np
from PIL import Image, ImageFile
import io
import wave
import matplotlib.pyplot as plt

# -------------------------------
# PIL Fix for truncated images
# -------------------------------
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------------------------
# Helper Functions
# -------------------------------
def image_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()

def bytes_to_image(b):
    return Image.open(io.BytesIO(b))

def encode_image_to_audio(image_bytes, audio_wave):
    audio_samples = np.frombuffer(audio_wave.readframes(audio_wave.getnframes()), dtype=np.int16)
    bin_image = ''.join([format(byte, '08b') for byte in image_bytes]) + '1111111111111110'
    if len(bin_image) > len(audio_samples):
        raise ValueError(f"Audio too short! Needed {len(bin_image)} samples, got {len(audio_samples)}")
    encoded_samples = audio_samples.copy()
    for i, bit in enumerate(bin_image):
        encoded_samples[i] = (encoded_samples[i] & ~1) | int(bit)
    return encoded_samples.tobytes()

def decode_image_from_audio(audio_wave):
    audio_samples = np.frombuffer(audio_wave.readframes(audio_wave.getnframes()), dtype=np.int16)
    bits = [str(sample & 1) for sample in audio_samples]
    bit_string = ''.join(bits)
    end_idx = bit_string.find('1111111111111110')
    if end_idx == -1:
        end_idx = len(bit_string)
    bit_string = bit_string[:end_idx]
    bit_string = bit_string[:len(bit_string) - (len(bit_string) % 8)]
    image_bytes = bytes([int(bit_string[i:i+8], 2) for i in range(0, len(bit_string), 8)])
    return bytes_to_image(image_bytes)

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

# -------------------------------
# WAV reader + Visualizations
# -------------------------------
def read_wav_samples_and_rate(file_like):
    """Return mono int16 samples and sample rate from a file-like WAV."""
    file_like.seek(0)
    with wave.open(file_like, 'rb') as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sampwidth = w.getsampwidth()
        n = w.getnframes()
        if sampwidth != 2:
            raise ValueError(f"Unsupported sample width: {sampwidth*8} bits. Please use 16-bit PCM WAV.")
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
    # Real FFT: positive frequencies only
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

# -------------------------------
# Streamlit Interface
# -------------------------------
st.title("Image-to-Audio Steganography Tool (Spectrogram + FFT)")
option = st.selectbox("Choose operation:", ["Encode", "Decode", "Detect"])

# -------------------------------
# Encode
# -------------------------------
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

        # Resize image if too big for audio
        max_bytes = num_samples // 8 - 2
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
            st.success("Image encoded into audio successfully!")

            # Visualizations of the encoded audio
            samples_enc, sr_enc = read_wav_samples_and_rate(encoded_audio_file)
            st.markdown("**Visualizations**")
            display_spectrogram(samples_enc, sr_enc, title="Encoded Audio Spectrogram")
            display_fft(samples_enc, sr_enc, title="Encoded Audio FFT Spectrum")

            encoded_audio_file.seek(0)
            st.download_button("Download Encoded Audio", encoded_audio_file, "encoded_audio.wav", "audio/wav")

        except Exception as e:
            st.error(f"Error: {e}")

# -------------------------------
# Decode
# -------------------------------
elif option == "Decode":
    st.subheader("Decode Image from Audio")
    audio_file = st.file_uploader("Upload Encoded WAV Audio", type=["wav"])

    if st.button("Decode") and audio_file:
        audio_file.seek(0)
        with wave.open(audio_file,'rb') as audio_wave:
            img = decode_image_from_audio(audio_wave)

        if img:
            st.image(img, caption="Decoded Image")
            samples_dec, sr_dec = read_wav_samples_and_rate(audio_file)
            st.markdown("**Visualizations**")
            display_spectrogram(samples_dec, sr_dec, title="Decoded Audio Spectrogram")
            display_fft(samples_dec, sr_dec, title="Decoded Audio FFT Spectrum")
        else:
            st.warning("No hidden image found!")

# -------------------------------
# Detect
# -------------------------------
elif option == "Detect":
    st.subheader("Detect Hidden Data in Audio")
    audio_file = st.file_uploader("Upload WAV Audio", type=["wav"])

    if st.button("Detect") and audio_file:
        audio_file.seek(0)
        with wave.open(audio_file,'rb') as audio_wave:
            hidden = detect_stego(audio_wave)

        samples_det, sr_det = read_wav_samples_and_rate(audio_file)
        st.markdown("**Visualizations**")
        display_spectrogram(samples_det, sr_det, title="Audio Spectrogram for Detection")
        display_fft(samples_det, sr_det, title="Audio FFT Spectrum for Detection")

        if hidden:
            st.success("Potential hidden data detected in audio!")
        else:
            st.info("No hidden data detected.")
