from __future__ import annotations
import wave
import numpy as np
import random
from typing import Tuple

HEADER_BITS = 32  # store payload length (in bits)

def _bits_from_bytes(b: bytes):
    for byte in b:
        for i in range(8):
            yield (byte >> (7 - i)) & 1

def _pack_length_bits(n: int):
    return [(n >> (HEADER_BITS - 1 - i)) & 1 for i in range(HEADER_BITS)]

def _read_wav_int16(path: str) -> Tuple[np.ndarray, int, int, int]:
    with wave.open(path, "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        fr = w.getframerate()
        n_frames = w.getnframes()
        if sampwidth != 2:
            raise ValueError("Only 16-bit PCM WAV is supported.")
        raw = w.readframes(n_frames)
    audio = np.frombuffer(raw, dtype=np.int16)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)
    return audio.copy(), n_channels, fr, n_frames

def _write_wav_int16(path: str, audio: np.ndarray, n_channels: int, fr: int):
    data = audio.astype(np.int16)
    if n_channels > 1:
        data = data.reshape(-1, n_channels)
    with wave.open(path, "wb") as w:
        w.setnchannels(n_channels)
        w.setsampwidth(2)
        w.setframerate(fr)
        w.writeframes(data.tobytes())

def embed_lsb_wav(in_wav_path: str, out_wav_path: str, message: str,
                  key: str = "audio", channel: int = 0, max_ratio: float | None = None):
    """
    Embed UTF-8 text in the LSBs of a WAV file.
    - key: scatters bit positions pseudo-randomly (harder to detect).
    - channel: which channel to use if stereo (0 = left, 1 = right).
    - max_ratio: optional cap on payload (0..1) relative to capacity to
      reduce detectability. If None, use full capacity.
    """
    audio, n_channels, fr, _ = _read_wav_int16(in_wav_path)
    samples = audio[:, channel] if n_channels > 1 else audio

    payload = message.encode("utf-8")
    payload_bits = list(_bits_from_bytes(payload))
    need = HEADER_BITS + len(payload_bits)
    capacity = samples.size

    if max_ratio is not None:
        capacity = int(capacity * float(max_ratio))

    if need > capacity:
        raise ValueError(f"Message too long. Need {need} bits, capacity {capacity} bits.")

    # keyed, pseudo-random scattering (stable across runs for same key)
    rng = random.Random(hash(key) & 0xFFFFFFFF)
    positions = list(range(samples.size))
    rng.shuffle(positions)
    use_pos = positions[:need]

    write_bits = _pack_length_bits(len(payload_bits)) + payload_bits
    stego = samples.copy()
    # set LSBs
    for pos, bit in zip(use_pos, write_bits):
        stego[pos] = (stego[pos] & ~1) | bit

    if n_channels > 1:
        audio[:, channel] = stego
        out = audio
    else:
        out = stego

    _write_wav_int16(out_wav_path, out, n_channels, fr)

def extract_lsb_wav(stego_wav_path: str, key: str = "audio", channel: int = 0) -> str:
    """Recover embedded UTF-8 text using the same key/channel."""
    audio, n_channels, fr, _ = _read_wav_int16(stego_wav_path)
    samples = audio[:, channel] if n_channels > 1 else audio

    rng = random.Random(hash(key) & 0xFFFFFFFF)
    positions = list(range(samples.size))
    rng.shuffle(positions)

    # header (payload length in bits)
    length = 0
    for p in positions[:HEADER_BITS]:
        length = (length << 1) | (samples[p] & 1)

    if length <= 0 or length > samples.size - HEADER_BITS:
        return ""

    # payload bits
    pay_positions = positions[HEADER_BITS:HEADER_BITS + length]
    bits = [(samples[p] & 1) for p in pay_positions]

    # pack to bytes
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i:i+8]:
            byte = (byte << 1) | b
        out.append(byte)

    try:
        return out.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        # fall back (message may be corrupted by attacks)
        return out.decode("utf-8", errors="ignore")
