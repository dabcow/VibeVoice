"""Background layers: real recordings mixed under the voice.

Every sound is a public-domain (CC0) field recording or effect from Freesound
(see assets/sounds/CREDITS.md), trimmed, looped and level-matched offline by
tools/build_sounds.py. A layer is one or more looping beds plus banks of
one-shot events that fire at random (Poisson) times, so it never repeats the
same way. When a layer is switched on its files are decoded once in a
background thread; playback then only mixes arrays, which costs almost no CPU.
"""
import json
import os
import sys
import threading

import numpy as np
import soundfile as sf

SR = 48000
TARGET_RMS = 10 ** (-24 / 20)  # loudness of every layer at volume 1
SOUNDS = os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "assets", "sounds")

# id, name, colour, description
LAYERS = [
    ("zoo", "Zoo", "#22c55e", "Elephants, lions, chimps, gibbons, sheep, cows, ducks and parrots over a busy zoo"),
    ("sea", "Sea", "#0ea5e9", "Big waves, seagulls and ship horns from a real harbour"),
    ("storm", "Storm", "#64748b", "Heavy rain, strong wind and real thunder strikes"),
    ("city", "City Traffic", "#f59e0b", "A busy city street, police and fire-truck sirens, horns and barking dogs"),
    ("jungle", "Jungle", "#16a34a", "Colombian and Mexican jungle, howler monkeys and frogs"),
    ("campfire", "Campfire", "#ea580c", "A crackling campfire, night crickets and owls"),
    ("construction", "Construction", "#eab308", "Jackhammers on concrete, hammering, machinery and reversing trucks"),
    ("space", "Space Battle", "#a855f7", "Laser blasts, explosions, alarms, missile-lock warnings and ship hum"),
    ("smoke", "Smoke Detector", "#e11d48", "A smoke alarm with a dying battery, chirping again and again"),
    ("firealarm", "Fire Alarm", "#dc2626", "A building fire alarm ringing nonstop, plus alarm announcements"),
    ("honking", "Honking", "#f97316", "Gridlocked traffic and endless car horns"),
    ("fastfood", "Fast Food Restaurant", "#facc15", "Lunch rush: chatter, orders called out at the counter, fryer beeps"),
    ("lecture", "University Lecture", "#6366f1", "A professor lecturing to a full hall, with the odd cough"),
    ("india", "Indian Street Market", "#d946ef", "Vendors calling out, tuk-tuks and horns on busy Indian streets"),
]
LAYER_IDS = [lay[0] for lay in LAYERS]

_manifest = None


def _load_manifest():
    global _manifest
    if _manifest is None:
        with open(os.path.join(SOUNDS, "manifest.json"), encoding="utf-8") as f:
            _manifest = json.load(f)
    return _manifest


def _read(name, as_int16=False):
    x, sr = sf.read(os.path.join(SOUNDS, name), dtype="int16" if as_int16 else "float32")
    if x.ndim > 1:
        x = x.mean(axis=1).astype(x.dtype)
    return x


# --------------------------------------------------------------------------
# banks (decoded once per layer, in the background)
# --------------------------------------------------------------------------
class Bank:
    def __init__(self, beds, events):
        self.beds = beds  # [(int16 loop, gain)]
        self.events = events  # [(int16 variants, events per minute, (gain lo, gain hi))]
        self.norm = 1.0


BANKS = {}
_building = set()
_lock = threading.Lock()


def build(layer_id):
    spec = _load_manifest()[layer_id]
    # 16-bit samples halve the memory; the 1/32768 scale is folded into the gains
    beds = [(_read(b["file"], as_int16=True), float(b["gain"]) / 32768.0) for b in spec["beds"]]
    events = [([_read(f, as_int16=True) for f in e["files"]], float(e["rate"]),
               (e["gain"][0] / 32768.0, e["gain"][1] / 32768.0)) for e in spec["events"]]
    bank = Bank(beds, events)
    # calibrate: every layer plays at the same loudness at volume 1,
    # without pushing sparse layers (a lone chirp) into clipping
    probe = LayerPlayer(SR, seed=1)
    sim = np.concatenate([probe.render_layer("probe", bank, 2048)
                          for _ in range(int(60 * SR / 2048))])
    rms = float(np.sqrt(np.mean(sim ** 2))) + 1e-9
    # worst case: two of the loudest one-shots overlapping at top gain, over the bed peaks
    bed_peak = sum(float(np.max(np.abs(b))) * g for b, g in beds)
    shot_peak = max([float(np.max(np.abs(v))) * ghi for vs, _rate, (_glo, ghi) in events
                     for v in vs] or [0.0])
    worst = max(float(np.max(np.abs(sim))), bed_peak + 1.5 * shot_peak) + 1e-9
    bank.norm = min(TARGET_RMS / rms, 0.9 / worst)
    return bank


def prepare(layer_id, on_ready=None):
    """Decode a layer's sounds in a background thread (no-op if ready or building)."""
    with _lock:
        if layer_id in BANKS or layer_id in _building:
            return
        _building.add(layer_id)

    def work():
        try:
            BANKS[layer_id] = build(layer_id)
        finally:
            with _lock:
                _building.discard(layer_id)
        if on_ready:
            on_ready(layer_id)
    threading.Thread(target=work, daemon=True).start()


def release(layer_id):
    BANKS.pop(layer_id, None)


def ready(layer_id):
    return layer_id in BANKS


# --------------------------------------------------------------------------
# real-time mixing
# --------------------------------------------------------------------------
class LayerPlayer:
    """Mixes active layers block by block: looping beds plus randomly fired events."""

    MAX_VOICES = 6

    def __init__(self, sr=SR, seed=None):
        self.sr = sr
        self.levels = {}  # layer id -> volume
        self.states = {}
        self.rng = np.random.default_rng(seed)

    def configure(self, levels):
        self.levels = {k: v for k, v in levels.items() if v > 0}

    @property
    def active(self):
        return bool(self.levels)

    def state_for(self, lid, bank):
        st = self.states.get(lid)
        if st is None or st["bank"] is not bank:
            st = {"bank": bank, "voices": [],
                  "pos": [int(self.rng.integers(len(b))) for b, _ in bank.beds]}
            self.states[lid] = st
            if bank.events:  # start with something happening
                self._fire(st, bank.events[int(self.rng.integers(len(bank.events)))], 0)
        return st

    def _fire(self, st, ev, offset):
        variants, _rate, (glo, ghi) = ev
        clip = variants[int(self.rng.integers(len(variants)))]
        st["voices"].append([clip, -offset, float(self.rng.uniform(glo, ghi))])

    def render_layer(self, lid, bank, n):
        st = self.state_for(lid, bank)
        out = np.zeros(n, np.float32)
        for j, (bed, g) in enumerate(bank.beds):
            p, L = st["pos"][j], len(bed)
            if p + n <= L:
                out += bed[p:p + n] * np.float32(g)
            else:
                k = L - p
                out[:k] += bed[p:] * np.float32(g)
                out[k:] += bed[:n - k] * np.float32(g)
            st["pos"][j] = (p + n) % L
        for ev in bank.events:
            if (len(st["voices"]) < self.MAX_VOICES
                    and self.rng.random() < ev[1] / 60 * n / self.sr):
                self._fire(st, ev, int(self.rng.integers(n)))
        keep = []
        for v in st["voices"]:
            clip, pos, g = v
            start, src = max(0, -pos), max(0, pos)
            m = min(n - start, len(clip) - src)
            if m > 0:
                out[start:start + m] += clip[src:src + m] * np.float32(g)
            v[1] = pos + n
            if v[1] < len(clip):
                keep.append(v)
        st["voices"] = keep
        return out

    def process(self, n):
        """Mixed background for the next n samples, or None when nothing is playing."""
        out = None
        for lid, vol in self.levels.items():
            bank = BANKS.get(lid)
            if bank is None:
                continue
            y = self.render_layer(lid, bank, n) * np.float32(vol * bank.norm)
            out = y if out is None else out + y
        return out
