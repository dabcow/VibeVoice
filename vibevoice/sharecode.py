"""Voice profiles <-> short share codes like "VV1-4fK9xQ...".

Layout (before base58): version, RGB colour, name, a bitmask of the
parameters that differ from the defaults, then one varint per set bit
(the value quantised to 1/20 of its slider step), and a 2-byte CRC so typos
are caught. Base58 has no punctuation, so chat apps can't mangle the code.
"""
import zlib

from .profiles import DEFAULTS, PARAMS

PREFIX = "VV1-"
VERSION = 1
FINE = 20  # quantise to slider resolution / FINE (keeps off-grid preset values)
B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
MAX_NAME = 28


def _b58encode(data):
    n = int.from_bytes(data, "big")
    out = ""
    while n:
        n, r = divmod(n, 58)
        out = B58[r] + out
    return "1" * (len(data) - len(data.lstrip(b"\0"))) + out


def _b58decode(text):
    n = 0
    for ch in text:
        i = B58.find(ch)
        if i < 0:
            raise ValueError(f"“{ch}” can't appear in a voice code")
        n = n * 58 + i
    zeros = len(text) - len(text.lstrip("1"))
    return b"\0" * zeros + (n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b"")


def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _read_varint(data, pos):
    n = shift = 0
    while True:
        if pos >= len(data):
            raise ValueError("This voice code is incomplete")
        b = data[pos]
        pos += 1
        n |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return n, pos


def encode(name, color, params):
    """Return a share code for a profile."""
    body = bytearray([VERSION])
    body += bytes.fromhex(color.lstrip("#")[:6].ljust(6, "0"))
    raw = name.strip()[:MAX_NAME].encode("utf-8")
    body += bytes([len(raw)]) + raw
    mask, values = 0, b""
    for i, (key, _label, lo, hi, res, _unit, _grp) in enumerate(PARAMS):
        v = min(hi, max(lo, float(params.get(key, DEFAULTS[key]))))
        q = round((v - lo) / (res / FINE))
        if q != round((DEFAULTS[key] - lo) / (res / FINE)):
            mask |= 1 << i
            values += _varint(q)
    body += _varint(mask) + values
    body += (zlib.crc32(bytes(body)) & 0xFFFF).to_bytes(2, "big")
    return PREFIX + _b58encode(bytes(body))


def decode(code):
    """Return {"name", "color", "params"} or raise ValueError with a friendly message."""
    text = "".join(code.split())  # tolerate spaces / line breaks from chat apps
    if text.upper().startswith(PREFIX.upper()):
        text = text[len(PREFIX):]
    elif text.upper().startswith("VV") and "-" in text[:5]:
        raise ValueError("This voice code is from a newer VibeVoice. Please update.")
    if not text:
        raise ValueError("Paste a voice code first")
    data = _b58decode(text)
    if len(data) < 7 or (zlib.crc32(data[:-2]) & 0xFFFF).to_bytes(2, "big") != data[-2:]:
        raise ValueError("That voice code isn't valid. Check it was copied completely.")
    data = data[:-2]
    if data[0] != VERSION:
        raise ValueError("This voice code is from a newer VibeVoice. Please update.")
    color = "#" + data[1:4].hex()
    ln = data[4]
    name = data[5:5 + ln].decode("utf-8", errors="replace").strip() or "Shared voice"
    mask, pos = _read_varint(data, 5 + ln)
    params = dict(DEFAULTS)
    for i, (key, _label, lo, hi, res, _unit, _grp) in enumerate(PARAMS):
        if mask >> i & 1:
            q, pos = _read_varint(data, pos)
            params[key] = round(min(hi, max(lo, lo + q * res / FINE)), 6)
    return {"name": name, "color": color, "params": params}
