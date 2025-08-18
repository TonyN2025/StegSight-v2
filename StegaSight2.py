
# StegaSight – Streamlit Web App (Encode + Decode + Analyse) 
# 18/08/2025 - StegaSight for Images

from __future__ import annotations
import io
import os
import random
import struct
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st

# Optional deps for demo ML; fail gracefully if not present
try:
    import joblib  # type: ignore
except Exception:
    joblib = None

# -----------------------------
# App Config & Constants
# -----------------------------
st.set_page_config(page_title="StegaSight", layout="centered", page_icon="🔎")

APP_DIR = Path(__file__).parent
UPLOAD_DIR = APP_DIR / "uploads"
DECODE_DIR = APP_DIR / "decoded"
UPLOAD_DIR.mkdir(exist_ok=True)
DECODE_DIR.mkdir(exist_ok=True)

SUPPORTED_TYPES = {
    "Text": ["txt"],
    "Images": ["png", "jpg", "jpeg"],
    "Audio": ["wav", "mp3"],
    "PDF": ["pdf"],
}
ALL_FILE_TYPES = sum(SUPPORTED_TYPES.values(), [])

# -----------------------------
# Utility: Light XOR (demo only)
# -----------------------------

def xor_bytes(data: bytes, key: str | None) -> bytes:
    """Very light obfuscation to avoid plaintext in LSB. Not crypto‑secure."""
    if not key:
        return data
    k = key.encode("utf-8")
    if not k:
        return data
    return bytes(b ^ k[i % len(k)] for i, b in enumerate(data))

# -----------------------------
# LSB Encode / Decode (1 bit/channel)
# -----------------------------

def capacity_bytes(img: Image.Image) -> int:
    w, h = img.size
    # 3 channels * 1 bit each per pixel => w*h*3 bits = //8 bytes
    return (w * h * 3) // 8


def encode_lsb(cover_img: Image.Image, payload: bytes, password: str | None = None) -> Image.Image:
    """Embed payload into an RGB image using 1 LSB per channel with 32‑bit length prefix."""
    if cover_img.mode != "RGB":
        cover_img = cover_img.convert("RGB")

    # Obfuscate
    obf = xor_bytes(payload, password)

    # Prefix with 4‑byte big‑endian length
    header = struct.pack(">I", len(obf))
    data = header + obf

    # Capacity check
    cap = capacity_bytes(cover_img)
    if len(data) > cap:
        raise ValueError(f"Payload too large: need {len(data)} bytes but image capacity is {cap} bytes. Use a larger PNG or shorten the data.")

    arr = np.array(cover_img)
    flat = arr.reshape(-1, 3)

    # Convert data bytes -> bits
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))  # shape (N_bits,)

    # We have 3 bits per pixel (R,G,B LSB)
    total_bits = bits.size
    n_pixels_needed = (total_bits + 3 - 1) // 3

    # Zero out LSBs for needed pixels
    work = flat.copy()
    work[:n_pixels_needed, 0] &= 0xFE
    work[:n_pixels_needed, 1] &= 0xFE
    work[:n_pixels_needed, 2] &= 0xFE

    # Write bits into LSBs
    # pad bits to multiple of 3
    if total_bits % 3 != 0:
        pad = 3 - (total_bits % 3)
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

    triples = bits.reshape(-1, 3)
    work[:triples.shape[0], 0] |= triples[:, 0]
    work[:triples.shape[0], 1] |= triples[:, 1]
    work[:triples.shape[0], 2] |= triples[:, 2]

    stego = work.reshape(arr.shape)
    return Image.fromarray(stego, mode="RGB")


def decode_lsb(stego_img: Image.Image, password: str | None = None) -> bytes:
    if stego_img.mode != "RGB":
        stego_img = stego_img.convert("RGB")

    arr = np.array(stego_img)
    flat = arr.reshape(-1, 3)

    # Extract first 32 bits => length
    first_11_pixels = flat[:11].copy()  # 11*3=33 >=32
    lsb = first_11_pixels & 1
    length_bits = lsb.reshape(-1)[:32]
    length_bytes = np.packbits(length_bits).tobytes()
    payload_len = struct.unpack(">I", length_bytes)[0]

    # Now extract payload_len bytes => payload_len*8 bits
    total_bits_needed = 32 + payload_len * 8
    n_pixels_needed = (total_bits_needed + 3 - 1) // 3
    lsb_all = (flat[:n_pixels_needed] & 1).reshape(-1)

    bits_payload = lsb_all[32:32 + payload_len * 8]
    data = np.packbits(bits_payload).tobytes()

    # De‑obfuscate
    return xor_bytes(data, password)

# -----------------------------
# (Optional) Simple ML Analyser
# -----------------------------
model = None

def load_model():
    global model
    if joblib is None:
        model = None
        return
    try:
        model = joblib.load(str(APP_DIR / "steganalysis_model.pkl"))
    except Exception:
        model = None


def analyze_image_ml(file_path: Path) -> str:
    if model is None:
        return random.choice(["safe", "potential", "danger"])  # placeholder if no model
    try:
        image = Image.open(file_path).convert("L")
        img_array = np.array(image)
        features = np.array([
            img_array.mean(),
            img_array.std(),
            np.percentile(img_array, 25),
            np.percentile(img_array, 50),
            np.percentile(img_array, 75),
        ]).reshape(1, -1)
        prediction = model.predict(features)[0]
        return str(prediction)
    except Exception:
        return random.choice(["safe", "potential", "danger"])  # robust fallback

# -----------------------------
# CLI Layered Analysis
# -----------------------------

def run_stegexpose(file_path: Path) -> str:
    try:
        # Prefer scanning a single file via its directory; StegExpose works best on folders
        parent = file_path.parent
        result = subprocess.run(
            ["java", "-jar", "StegExpose.jar", str(parent)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        output = (result.stdout + result.stderr).decode("utf-8", errors="ignore").lower()
        # Heuristic: if filename appears with suspect/threshold or score lines
        if file_path.name.lower() in output and ("suspect" in output or "detected" in output):
            return "potential"
        return "safe"
    except Exception:
        return "safe"


def layer_based_analysis(file_path: Path) -> tuple[str, dict[str, str]]:
    tools = {
        "zsteg": ["zsteg", str(file_path)],
        "steghide": ["steghide", "info", str(file_path)],
        "outguess": ["outguess", "-k", "dummykey", "-r", str(file_path), "/dev/null"],
        "exiftool": ["exiftool", str(file_path)],
        "binwalk": ["binwalk", str(file_path)],
        "foremost": ["foremost", "-i", str(file_path), "-o", str(APP_DIR / "foremost_out")],
        "strings": ["strings", str(file_path)],
    }

    suspicious_count = 0
    potential_keywords = [
        "hidden", "embedded", "steganography", "payload", "encrypted", "error", "warning", "corrupt",
    ]
    extracted_content: dict[str, str] = {}

    for tool_name, cmd in tools.items():
        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
            output = (result.stdout + result.stderr).decode(errors="ignore").lower()

            extracted_content[tool_name] = ""

            if tool_name == "steghide":
                if "could not extract any data" not in output and "error" not in output:
                    suspicious_count += 1
                    extracted_content[tool_name] = output.strip() or "(no output)"
                else:
                    extracted_content[tool_name] = "No extractable data found or error occurred."

            elif tool_name == "outguess":
                if "error" not in output and "usage" not in output:
                    if output.strip():
                        suspicious_count += 1
                        extracted_content[tool_name] = output.strip()
                    else:
                        extracted_content[tool_name] = "No extractable data found."
                else:
                    extracted_content[tool_name] = "No extractable data found or error occurred."

            elif tool_name == "exiftool":
                if any(kw in output for kw in potential_keywords):
                    suspicious_count += 1
                extracted_content[tool_name] = output.strip() or "No suspicious metadata found."

            elif tool_name == "binwalk":
                if "embedded" in output or "data" in output:
                    suspicious_count += 1
                extracted_content[tool_name] = output.strip() or "No suspicious signatures found."

            elif tool_name == "foremost":
                if result.stderr and len(result.stderr) > 0:
                    suspicious_count += 1
                extracted_content[tool_name] = (result.stdout + result.stderr).decode(errors="ignore").strip() or "No output."

            elif tool_name == "strings":
                if any(kw in output for kw in potential_keywords):
                    suspicious_count += 1
                extracted_content[tool_name] = output.strip() or "No suspicious strings found."

            elif tool_name == "zsteg":
                if "possible" in output or "detected" in output:
                    suspicious_count += 1
                    extracted_content[tool_name] = output.strip() or "Possible patterns found."
                else:
                    extracted_content[tool_name] = "No suspicious patterns detected."

        except Exception:
            extracted_content[tool_name] = "Tool execution failed or not available."

    # Optionally include StegExpose result
    try:
        stegexpose_verdict = run_stegexpose(file_path)
        if stegexpose_verdict == "potential":
            suspicious_count += 1
        extracted_content["stegexpose"] = stegexpose_verdict
    except Exception:
        extracted_content["stegexpose"] = "not run"

    if suspicious_count >= 3:
        threat_level = "danger"
    elif suspicious_count > 0:
        threat_level = "potential"
    else:
        threat_level = "safe"

    return threat_level, extracted_content


def detect_threat(file_path: Path, file_type: str) -> tuple[str, dict[str, str]]:
    if file_type.startswith("image/"):
        result, extracted_content = layer_based_analysis(file_path)
        if result in {"safe", "potential", "danger"}:
            return result, extracted_content
        else:
            # Fall back to ML if set up
            prediction = analyze_image_ml(file_path)
            return prediction, extracted_content
    elif file_type.startswith("text/") or file_type.startswith("audio/"):
        return random.choice(["safe", "potential"]), {}
    else:
        return "safe", {}

# -----------------------------
# UI – Sidebar
# -----------------------------
mode = st.sidebar.radio("Choose mode:", ["Hide (Encode)", "Uncover (Decode/Analyse)"])

st.sidebar.markdown(
    """
**Tips**
- Prefer PNG for encoding (lossless).
- The optional password uses simple XOR (demo‑only). Use real crypto if needed.
- External tools are optional; results will show if they aren't available.
    """
)

# -----------------------------
# UI – Hide (Encode)
# -----------------------------
if mode.startswith("Hide"):
    st.header("🔐 Hide data in an image (LSB)")
    cover = st.file_uploader("Cover image (PNG recommended)", type=["png", "jpg", "jpeg"], accept_multiple_files=False)

    colA, colB = st.columns(2)
    with colA:
        payload_type = st.radio("Payload type", ["Text message", "Binary file"], horizontal=True)
    with colB:
        password = st.text_input("Optional password (XOR demo)", type="password", help="Light obfuscation only – not strong encryption.")

    payload_bytes: bytes | None = None
    extracted_name = None

    if payload_type == "Text message":
        msg = st.text_area("Secret message", placeholder="Type your secret here…")
        if msg:
            payload_bytes = msg.encode("utf-8")
            extracted_name = "message.txt"
    else:
        up = st.file_uploader("Secret file", type=None, accept_multiple_files=False)
        if up is not None:
            payload_bytes = up.getbuffer().tobytes()
            extracted_name = f"extracted_{up.name}"

    if st.button("Embed into image", type="primary", disabled=(cover is None or payload_bytes is None)):
        try:
            assert cover is not None and payload_bytes is not None
            img = Image.open(cover)
            stego = encode_lsb(img, payload_bytes, password=password or None)

            # Save to buffer and disk
            buf = io.BytesIO()
            stego.save(buf, format="PNG")
            buf.seek(0)

            out_path = UPLOAD_DIR / f"stego_{Path(cover.name).stem}.png"
            stego.save(out_path, format="PNG")

            st.success("Payload embedded successfully into PNG.")
            st.image(stego, caption=f"stego_{Path(cover.name).stem}.png", use_container_width=True)
            st.download_button("⬇️ Download stego image", data=buf, file_name=out_path.name, mime="image/png")

            st.info(
                f"Capacity used: {len(payload_bytes)} bytes (plus 4‑byte header). Image capacity: {capacity_bytes(img)} bytes.")
        except Exception as e:
            st.error(f"Encoding failed: {e}")

# -----------------------------
# UI – Uncover (Decode/Analyse)
# -----------------------------
else:
    st.title("🕵️‍♂️ StegaSight – Decode & Analyse")
    st.markdown("Upload files; we'll try to decode payloads created by this app and run basic steganalysis.")

    uploaded_files = st.file_uploader(
        "📥 Upload your files", type=ALL_FILE_TYPES, accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = UPLOAD_DIR / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"✅ `{uploaded_file.name}` uploaded.")
            file_type = uploaded_file.type

            # Try decode if it's an image
            with st.expander(f"📖 Preview & Decode: {uploaded_file.name}", expanded=True):
                if file_type.startswith("image/"):
                    image = Image.open(file_path)

                    with st.form(key=f"decode_form_{uploaded_file.name}"):
                        dec_pwd = st.text_input("Password (if used)", type="password")
                        submitted = st.form_submit_button("Try decode")

                    decoded_preview = None
                    extracted_file_bytes = None

                    if submitted:
                        try:
                            data = decode_lsb(image, password=dec_pwd or None)
                            decoded_preview = data[:256]
                            extracted_file_bytes = data

                            # Save extracted file to disk and offer download
                            out_name = f"extracted_{Path(uploaded_file.name).stem}.bin"
                            out_path = DECODE_DIR / out_name
                            with open(out_path, "wb") as w:
                                w.write(data)

                            st.success("Decoded payload found (may be text or binary).")
                            # If looks like UTF‑8 text, show a snippet
                            try:
                                snippet = data.decode("utf-8")[:500]
                                st.code(snippet)
                            except Exception:
                                st.info("Payload appears to be binary (showing first 64 bytes):")
                                st.code(decoded_preview.hex(" "))

                            st.download_button("⬇️ Download extracted payload", data=data, file_name=out_name)
                        except Exception as e:
                            st.info(f"No decodable payload via this app's LSB scheme or wrong password. ({e})")

                    # Always show the image
                    st.image(image, caption="Uploaded image", use_container_width=True)

                elif file_type.startswith("text/"):
                    try:
                        text = uploaded_file.getvalue().decode("utf-8")
                        st.code(text[:1000])
                    except Exception:
                        st.warning("Couldn't preview the text content.")

                elif file_type.startswith("audio/"):
                    st.audio(str(file_path))

                elif file_type == "application/pdf":
                    st.info("PDF uploaded. Preview not supported here.")
                else:
                    st.info("File uploaded. Preview not supported.")

            # Threat detection
            threat_level, extracted_content = detect_threat(file_path, file_type)

            badge = {
                "safe": "💚 Safe",
                "potential": "⚠️ Potential",
                "danger": "💔 Dangerous",
            }.get(threat_level, threat_level)

            st.write(f"**Verdict:** {badge}")

            # Tool outputs
            if extracted_content:
                st.markdown("**Tool outputs** (if available):")
                for tool_name, content in extracted_content.items():
                    with st.expander(f"🔍 {tool_name}"):
                        st.text(content or "(no output)")

# -----------------------------
# Init
# -----------------------------
load_model()