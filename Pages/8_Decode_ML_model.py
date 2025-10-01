import streamlit as st      # web-interface
from PIL import Image       # importing images
from pathlib import Path    # for path operations
from stegano import lsb     # Python Steganography module
import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as F


st.set_page_config(page_title="Uncovering", page_icon="🔎")

st.title("🔎 Find what's hidden")

uploaded_files = st.file_uploader("Upload your image file", type=["png"], accept_multiple_files=True)

uploads_dir = Path("uploads_ML")
uploads_dir.mkdir(exist_ok=True)

# ML Model Setup Stuff

class TopLeftCrop:
    def __init__(self, size):
            self.size = size
    def __call__(self, img):
        height = self.size
        if img.height > 100:
            return F.crop(img, top=0, left=0, height=height, width=img.width)
        else:
            return(img)

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()


        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

class LSBHighlight3(object):
    def __call__(self, img):
        img2 = img.convert("RGB")
        width, height = img2.size

        lsb_image = Image.new('RGB', (width, height))

        # Load pixels for faster access
        original_pixels = img2.load()
        lsb_pixels = lsb_image.load()

        for x in range(width):
            for y in range(height):
                # Get the RGB tuple for the pixel at (x, y)
                r, g, b = original_pixels[x, y]

                # Isolate the LSB of each color channel
                lsb_r = r & 1
                lsb_g = g & 1
                lsb_b = b & 1

                # Scale each LSB value (0 or 1) to a full 0 or 255
                lsb_r_scaled = lsb_r * 255
                lsb_g_scaled = lsb_g * 255
                lsb_b_scaled = lsb_b * 255

                # Set the pixel in the new LSB image with the scaled LSB values
                lsb_pixels[x, y] = (lsb_r_scaled, lsb_g_scaled, lsb_b_scaled)

        return lsb_image

transform = transforms.Compose([
    TopLeftCrop(100),
    LSBHighlight3(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = 'Models/model_lsb_2.pth'

try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
except Exception as e:
    print(f"Failed to load checkpoint: {e}")
    sys.exit(1)

class_to_idx = checkpoint.get('class_to_idx')
if class_to_idx is None:
    print("Checkpoint does not contain 'class_to_idx'.")
    sys.exit(1)

idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes = len(idx_to_class)

model = SimpleCNN(num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

cols = st.columns(2)
cols[0].header("Clean")
cols[1].header("Hidden")

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

        #with st.expander(f"Preview of {uploaded_file.name}", expanded=True):
            st.image(img, use_container_width=True)

        input_tensor = transform(img).unsqueeze(0).to(device)

        hidden = True
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            class_index = predicted.item()
            class_label = idx_to_class[class_index]
            imgCol = Image.open(file_path)
            if class_index == 0:
                Hidden = False
                cols[0].image(imgCol)
                #cols[0].warning("No hidden message to decrypt.")
            else:
                cols[1].image(imgCol)
                cols[1].text(uploaded_file.name)
                try:
                    display_text = lsb.reveal(img)
                    st.markdown(
                        f"Extracted hidden message of <span style='color:#1E90FF;'>{uploaded_file.name}",
                        unsafe_allow_html=True
                    )
                    st.text_area("", display_text, height=200, key=uploaded_file)
                except IndexError:
                    print("False positive detected")
                    cols[1].error(f"False positive detected")


















