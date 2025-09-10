from PIL import Image
import numpy as np
import os
import argparse

def message_to_bits(message):
    return ''.join([format(ord(char), '08b') for char in message])

def encode_image(input_image_path, output_image_path, secret_message):
    image = Image.open(input_image_path)
    image = image.convert('RGB')
    np_image = np.array(image).astype(np.uint8)  # Make sure it's uint8

    message_bits = message_to_bits(secret_message) + '1111111111111110'  # End delimiter

    h, w, _ = np_image.shape
    flat_image = np_image.reshape(-1, 3)

    bit_index = 0
    for i in range(flat_image.shape[0]):
        for j in range(3):  # For each color channel (R, G, B)
            if bit_index < len(message_bits):
                # Use 254 mask instead of ~1 to avoid negative overflow
                flat_image[i][j] = (flat_image[i][j] & 254) | int(message_bits[bit_index])
                bit_index += 1

    if bit_index < len(message_bits):
        raise ValueError("Message is too long to encode in the image!")

    encoded_image = flat_image.reshape((h, w, 3))
    encoded_image_pil = Image.fromarray(encoded_image.astype('uint8'))
    encoded_image_pil.save(output_image_path)

    print("✅ Encoding complete. Output saved to:", output_image_path)
    print("Full path:", os.path.abspath(output_image_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode a secret message into an image.")
    parser.add_argument("-i", "--input", required=True, help="Input cover image file path")
    parser.add_argument("-o", "--output", required=True, help="Output encoded image file path")
    parser.add_argument("-s", "--secret", required=True, help="Secret message to encode")

    args = parser.parse_args()

    encode_image(args.input, args.output, args.secret)

