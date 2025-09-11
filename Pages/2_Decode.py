import streamlit as st      # web-interface
from PIL import Image       # importing images
from pathlib import Path    # for path operations
from stegano import lsb     # Python Steganography module
import numpy as np          # Working with area
import hashlib
import base64
from cryptography.fernet import Fernet, InvalidToken

st.set_page_config(page_title="Uncovering", page_icon="🔎")

st.title("🔎 Find what's hidden")

uploaded_files = st.file_uploader("Upload your image file", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

password = st.text_input("Enter password (if used)", type="password")

brute_force_file_btn = st.button("Brute-Force Using Password File")

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = uploads_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            img = Image.open(file_path)
        except Exception as e:
            st.error(f"Error opening image: {e}")
            continue

        with st.expander(f"Preview of {uploaded_file.name}", expanded=True):
            st.image(img, use_container_width=True)

        hidden_text = None  
        try:
            hidden_text = lsb.reveal(img)
        except Exception as e:
            error_message = str(e)
            if "Impossible to detect message" not in error_message:
                st.error(f"Error decoding hidden message: {e}")

        # Always attempt to display the extracted string, even if clean
        if hidden_text is None:
            st.markdown(
                f"No hidden message detected. Extracted string: ''",
                unsafe_allow_html=True
            )
        else:
            # Always show the raw message preview (first 500 chars)
            display_text = hidden_text
            decrypted_text = None
            decryption_attempted = False
            decrypt_btn = st.button("Decrypt Hidden Message", key=f"decrypt-btn-{uploaded_file.name}")
            if decrypt_btn:
                if not password:
                    st.warning("A password is required to attempt decryption.")
                else:
                    decryption_attempted = True
                    # Derive key from password
                    key = base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())
                    fernet = Fernet(key)
                    try:
                        decrypted_bytes = fernet.decrypt(hidden_text.encode())
                        decrypted_text = decrypted_bytes.decode()
                        display_text = decrypted_text
                    except (InvalidToken, ValueError, Exception):
                        st.warning("Failed to decrypt the hidden message with the provided password. Displaying raw extracted message below.")
                        display_text = hidden_text

            # Brute-force decryption section
            st.markdown("---")
            st.markdown("### Brute-Force Decrypt Hidden Message")
            # Removed the text area input and brute-force button that uses passwords_input

            # New brute-force using password file button handling
            if brute_force_file_btn:
                if not hidden_text:
                    st.warning("No hidden message to decrypt.")
                else:
                    pwd_file_path = Path("passwords.txt")
                    if not pwd_file_path.exists():
                        st.warning("Password file 'passwords.txt' not found.")
                    else:
                        try:
                            with open(pwd_file_path, "r", encoding="utf-8") as pf:
                                file_passwords = [line.strip() for line in pf if line.strip()]
                            if not file_passwords:
                                st.warning("Password file 'passwords.txt' is empty.")
                            else:
                                success = False
                                for pwd in file_passwords:
                                    try:
                                        key = base64.urlsafe_b64encode(hashlib.sha256(pwd.encode()).digest())
                                        fernet = Fernet(key)
                                        decrypted_bytes = fernet.decrypt(hidden_text.encode())
                                        decrypted_message = decrypted_bytes.decode()
                                        st.success(f"Successfully decrypted with password from file: **{pwd}**")
                                        st.text_area("Decrypted message:", decrypted_message, height=200)
                                        success = True
                                        break
                                    except (InvalidToken, ValueError, Exception):
                                        continue
                                if not success:
                                    st.warning("Brute-force decryption failed with all passwords in 'passwords.txt'.")
                        except Exception as e:
                            st.error(f"Error reading password file: {e}")

            # Display preview (first 500 chars)
            display_text = display_text[:500]
            st.markdown(
                f"Extracted hidden message of <span style='color:#1E90FF;'>{uploaded_file.name}</span> (first 500 characters):",
                unsafe_allow_html=True
            )
            st.text_area("", display_text, height=200)

        # Re-open the image to avoid "Operation on closed image" error after lsb.reveal
        try:
            img = Image.open(file_path)
        except Exception as e:
            st.error(f"Error re-opening image for pixel analysis: {e}")
            continue

        # Show pixel values in binary and decimal for pixels in selected block
        pixels = np.array(img)
        # Flatten pixels to a 2D array with shape (num_pixels, channels)
        if pixels.ndim == 2:  # Grayscale image
            flat_pixels = pixels.flatten()
            channels = 1
        else:
            flat_pixels = pixels.reshape(-1, pixels.shape[-1])
            channels = pixels.shape[-1]

        total_pixels = len(flat_pixels)
        max_block_index = (total_pixels - 1) // 100
        block_index = st.number_input(
            f"Select pixel block to view for {uploaded_file.name} (0-based index)",
            min_value=0,
            max_value=max_block_index,
            value=0,
            step=1
        )

        start = block_index * 100
        end = start + 100
        display_pixels = flat_pixels[start:end]

        # Prepare data for display
        binary_values = []
        decimal_values = []
        for px in display_pixels:
            if channels == 1:
                decimal_values.append(str(px))
                binary_values.append(format(px, '08b'))
            else:
                decimal_str = ','.join(str(v) for v in px)
                binary_str = ','.join(format(v, '08b') for v in px)
                decimal_values.append(decimal_str)
                binary_values.append(binary_str)

        # Combine into a table with pixel numbers as top row and "Binary"/"Decimal" as first column
        table_header = "<table><thead><tr><th>Pixel</th>"
        for i in range(len(display_pixels)):
            table_header += f"<th>{start + i + 1}</th>"
        table_header += "</tr></thead><tbody>"

        binary_row = "<tr><td>Binary</td>"
        for binv in binary_values:
            binary_row += f"<td>{binv}</td>"
        binary_row += "</tr>"

        decimal_row = "<tr><td>Decimal</td>"
        for dec in decimal_values:
            decimal_row += f"<td>{dec}</td>"
        decimal_row += "</tr>"

        table_str = table_header + binary_row + decimal_row + "</tbody></table>"

        wrapped_table = f'<div style="overflow-x: auto; white-space: nowrap;">{table_str}</div>'

        st.markdown(f"Pixel values (pixels {start + 1}–{min(end, total_pixels)} of {total_pixels}):")
        st.markdown(wrapped_table, unsafe_allow_html=True)
        st.markdown("*Note: Only 100 pixels are shown per block to improve performance. You can select different blocks to view more pixels.*")
