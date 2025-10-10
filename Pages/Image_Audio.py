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
    return lsb_ratio > 0.45 and lsb_ratio < 0.55

def save_encoded_wav(encoded_bytes, params):
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as w:
        w.setparams(params)
        w.writeframes(encoded_bytes)
    buf.seek(0)
    return buf

def plot_waveform(audio_samples, title="Waveform"):
    plt.figure(figsize=(10, 3))
    plt.plot(audio_samples, color='blue')
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    st.pyplot(plt)
    plt.close()

# -------------------------------
# Spectrogram + Heatmap Analysis
# -------------------------------
def compute_suspiciousness(Pxx, freqs):
    high_freq = freqs >= 8000
    high_std = np.std(Pxx[high_freq, :], axis=0)
    uniform_score = 1 - (high_std / np.max(high_std))
    var_per_bin = np.var(Pxx, axis=0)
    var_score = 1 - (var_per_bin / np.max(var_per_bin))
    suspiciousness = (uniform_score + var_score) / 2
    return suspiciousness

def spectrogram_heatmap(audio_samples, framerate):
    plt.figure(figsize=(10, 4))
    Pxx, freqs, bins, im = plt.specgram(audio_samples, Fs=framerate, cmap='viridis')
    suspiciousness = compute_suspiciousness(Pxx, freqs)
    heatmap = np.tile(suspiciousness, (len(freqs), 1))
    plt.imshow(heatmap, aspect='auto', origin='lower', extent=[bins[0], bins[-1], freqs[0], freqs[-1]],
               cmap='Reds', alpha=0.5)
    plt.title("Spectrogram with Suspiciousness Heatmap")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    st.pyplot(plt)
    plt.close()
    suspicious_percent = 100 * np.mean(suspiciousness)
    return suspicious_percent, Pxx, freqs

def analyze_frequency_bands(Pxx, freqs):
    low = freqs < 3000
    mid = (freqs >= 3000) & (freqs < 8000)
    high = freqs >= 8000
    low_std = np.std(Pxx[low, :])
    mid_std = np.std(Pxx[mid, :])
    high_std = np.std(Pxx[high, :])
    low_energy = np.mean(Pxx[low, :])
    mid_energy = np.mean(Pxx[mid, :])
    high_energy = np.mean(Pxx[high, :])
    total_energy = low_energy + mid_energy + high_energy
    high_energy_ratio = high_energy / total_energy
    var_per_bin = np.var(Pxx, axis=0)
    uniform_bins = var_per_bin < np.median(var_per_bin) * 0.8
    uniform_percent = 100 * np.sum(uniform_bins) / len(uniform_bins)
    report = {
        "Suspicious bins (%)": 100*np.mean(var_per_bin < np.median(var_per_bin)*0.8),
        "High-frequency std": high_std,
        "Low-frequency std": low_std,
        "Mid-frequency std": mid_std,
        "High-frequency energy ratio": high_energy_ratio,
        "Uniform time bins (%)": uniform_percent
    }
    return report

def display_spectrogram_report(report, suspicious_percent):
    st.subheader("Spectrogram Analysis Report")
    st.write(f"Suspicious high-frequency regions: {report['Suspicious bins (%)']:.2f}%")
    st.write(f"High-frequency standard deviation: {report['High-frequency std']:.2f} (low may indicate hidden data)")
    st.write(f"Low-frequency std: {report['Low-frequency std']:.2f}")
    st.write(f"Mid-frequency std: {report['Mid-frequency std']:.2f}")
    st.write(f"High-frequency energy ratio: {report['High-frequency energy ratio']:.2f}")
    st.write(f"Uniform time bins: {report['Uniform time bins (%)']:.2f}%")
    st.info(
        "Audio is suspicious if high-frequency regions are unusually uniform or have elevated energy "
        "compared to low/mid frequencies. This often indicates potential LSB steganography."
    )
    st.progress(min(int(suspicious_percent), 100))

# -------------------------------
# Streamlit Interface
# -------------------------------
st.title("Image-to-Audio Steganography Tool with Heatmap Spectrogram Analysis")
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

        # Resize if too big
        max_bytes = num_samples // 8 - 2
        if len(image_bytes) > max_bytes:
            scale_factor = (max_bytes / len(image_bytes))**0.5
            new_size = (int(img.width*scale_factor), int(img.height*scale_factor))
            img = img.resize(new_size, Image.ANTIALIAS)
            image_bytes = image_to_bytes(img)
            st.warning(f"Image resized to {new_size} to fit audio length.")

        try:
            # Encode
            encoded_audio_bytes = encode_image_to_audio(image_bytes, audio_wave)
            audio_params = audio_wave.getparams()
            audio_wave.close()

            # Save encoded audio
            encoded_audio_file = save_encoded_wav(encoded_audio_bytes, audio_params)
            encoded_audio_file.seek(0)

            st.success("Image encoded into audio successfully!")

            # Original waveform
            audio_file.seek(0)
            with wave.open(audio_file, 'rb') as audio_wave_orig:
                samples_orig = np.frombuffer(audio_wave_orig.readframes(audio_wave_orig.getnframes()), dtype=np.int16)
            plot_waveform(samples_orig, "Original Audio Waveform")

            # Encoded waveform
            encoded_audio_file.seek(0)
            with wave.open(encoded_audio_file, 'rb') as audio_wave_enc:
                samples_enc = np.frombuffer(audio_wave_enc.readframes(audio_wave_enc.getnframes()), dtype=np.int16)
            plot_waveform(samples_enc, "Encoded Audio Waveform")

            # Download
            encoded_audio_file.seek(0)
            st.download_button("Download Encoded Audio", encoded_audio_file, "encoded_audio.wav", "audio/wav")

        except Exception as e:
            st.error(f"Error: {e}")

# -------------------------------
# Decode
# -------------------------------
elif option=="Decode":
    st.subheader("Decode Image from Audio")
    audio_file = st.file_uploader("Upload Encoded WAV Audio", type=["wav"])

    if st.button("Decode") and audio_file:
        audio_file.seek(0)
        with wave.open(audio_file,'rb') as audio_wave:
            img = decode_image_from_audio(audio_wave)

        if img:
            st.image(img, caption="Decoded Image")

            # Waveform preview
            audio_file.seek(0)
            with wave.open(audio_file,'rb') as audio_wave:
                audio_samples = np.frombuffer(audio_wave.readframes(audio_wave.getnframes()), dtype=np.int16)
            plot_waveform(audio_samples,"Decoded Audio Waveform")

            # Spectrogram heatmap & report
            suspicious_percent, Pxx, freqs = spectrogram_heatmap(audio_samples, audio_wave.getframerate())
            report = analyze_frequency_bands(Pxx, freqs)
            display_spectrogram_report(report, suspicious_percent)
        else:
            st.warning("No hidden image found!")

# -------------------------------
# Detect
# -------------------------------
elif option=="Detect":
    st.subheader("Detect Hidden Data in Audio")
    audio_file = st.file_uploader("Upload WAV Audio", type=["wav"])

    if st.button("Detect") and audio_file:
        audio_file.seek(0)
        with wave.open(audio_file,'rb') as audio_wave:
            hidden = detect_stego(audio_wave)

        # Waveform preview
        audio_file.seek(0)
        with wave.open(audio_file,'rb') as audio_wave:
            audio_samples = np.frombuffer(audio_wave.readframes(audio_wave.getnframes()), dtype=np.int16)
        plot_waveform(audio_samples,"Audio Waveform for Detection")

        # Spectrogram heatmap & report
        suspicious_percent, Pxx, freqs = spectrogram_heatmap(audio_samples,audio_wave.getframerate())
        report = analyze_frequency_bands(Pxx,freqs)
        display_spectrogram_report(report, suspicious_percent)

        if hidden:
            st.success("Potential hidden data detected in audio!")
        else:

            st.info("No hidden data detected.")
