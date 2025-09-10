from PIL import Image
import numpy as np
import argparse

def bits_to_message(bits):
    chars = []
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        char = chr(int(byte, 2))
        chars.append(char)
    return ''.join(chars)

def decode_image(encoded_image_path):
    image = Image.open(encoded_image_path)
    image = image.convert('RGB')
    np_image = np.array(image)

    h, w, _ = np_image.shape
    flat_image = np_image.reshape(-1, 3)

    bits = []
    for i in range(flat_image.shape[0]):
        for j in range(3):  # R, G, B channels
            bits.append(str(flat_image[i][j] & 1))

    bits = ''.join(bits)
    # Find the end delimiter '1111111111111110'
    end_index = bits.find('1111111111111110')
    if end_index == -1:
        raise ValueError("No hidden message found!")

    message_bits = bits[:end_index]
    message = bits_to_message(message_bits)
    return message

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode secret message from encoded image.")
    parser.add_argument("-i", "--input", required=True, help="Input encoded image file path")

    args = parser.parse_args()

    secret_message = decode_image(args.input)
    print("🔓 Decoded secret message:")
    print(secret_message)

