import streamlit as st      #web-interface
from PIL import Image       #for importing images
from pathlib import Path    #for path operations
from stegano import lsb
import numpy as np

st.set_page_config(page_title="Uncovering", page_icon="🔎")

st.title("🔎 Find what's hidden")

uploaded_files = st.file_uploader("Upload your image file", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)

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

        try:
            hidden_text = lsb.reveal(img)
            if not hidden_text:
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
                continue
        except Exception as e:
            error_message = str(e)
            if "Impossible to detect message" in error_message:
                # Treat as no hidden message found, show pixel values for pixels in selected block only
                pixels = np.array(img)
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
                st.markdown("*Note: Only 100 pixels are shown per block to improve performance.*")
                continue
            else:
                st.error(f"Error decoding hidden message: {e}")
                continue

        display_text = hidden_text[:500]
        st.markdown(
            f"Extracted hidden message of <span style='color:#1E90FF;'>{uploaded_file.name}</span> (first 500 characters):",
            unsafe_allow_html=True
        )
        st.text_area("", display_text, height=200)
