"""Draw the VibeVoice icon (gradient badge + sound bars) to assets/icon.png and .ico.

Pure numpy + zlib, so no imaging library is needed. Run: python tools/make_icon.py
"""
import os
import struct
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
TOP, BOTTOM = np.array([124, 92, 255]), np.array([236, 72, 153])  # purple -> pink
BARS = (0.34, 0.62, 0.92, 0.62, 0.34)  # relative bar heights


def render(size, ss=4):
    n = size * ss
    y, x = (np.mgrid[0:n, 0:n] + 0.5) / n  # 0..1 coordinates
    # rounded square badge
    r, m = 0.22, 0.04
    qx = np.maximum(np.abs(x - 0.5) - (0.5 - m - r), 0)
    qy = np.maximum(np.abs(y - 0.5) - (0.5 - m - r), 0)
    badge = np.hypot(qx, qy) <= r
    t = np.clip((x + y) / 2, 0, 1)[..., None]
    rgb = TOP * (1 - t) + BOTTOM * t
    # white rounded sound bars
    bars = np.zeros_like(badge)
    w, gap = 0.085, 0.045
    x0 = 0.5 - (len(BARS) * w + (len(BARS) - 1) * gap) / 2
    for i, h in enumerate(BARS):
        cx = x0 + i * (w + gap) + w / 2
        half = h * 0.30
        dy = np.maximum(np.abs(y - 0.5) - (half - w / 2), 0)
        bars |= np.hypot(x - cx, dy) <= w / 2
    rgb = np.where(bars[..., None], 255.0, rgb)
    alpha = badge.astype(float) * 255
    img = np.dstack((rgb, alpha)).reshape(size, ss, size, ss, 4).mean(axis=(1, 3))
    return img.round().astype(np.uint8)


def png_bytes(img):
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[row].tobytes() for row in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def main():
    os.makedirs(ASSETS, exist_ok=True)
    sizes = (256, 64, 48, 32, 24, 16)
    pngs = [png_bytes(render(s)) for s in sizes]
    with open(os.path.join(ASSETS, "icon.png"), "wb") as f:
        f.write(pngs[0])
    # ICO container with PNG-compressed entries (supported since Windows Vista)
    header = struct.pack("<HHH", 0, 1, len(sizes))
    offset = 6 + 16 * len(sizes)
    entries, blobs = b"", b""
    for s, data in zip(sizes, pngs):
        entries += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
        blobs += data
    with open(os.path.join(ASSETS, "icon.ico"), "wb") as f:
        f.write(header + entries + blobs)
    print("wrote", os.path.join(ASSETS, "icon.png"), "and icon.ico")


if __name__ == "__main__":
    main()
