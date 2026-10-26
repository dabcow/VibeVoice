"""Parameter schema and built-in voice profiles."""

# key, label, min, max, resolution, unit, group
PARAMS = [
    ("pitch", "Pitch", -12, 12, 0.5, "st", "Pitch"),
    ("formant", "Formant (timbre)", -12, 12, 0.5, "st", "Pitch"),
    ("tremble", "Tremble (shaky)", 0, 1, 0.01, "", "Pitch"),
    ("robot_mix", "Monotone robot", 0, 1, 0.01, "", "Character"),
    ("robot_pitch", "Robot pitch", 50, 400, 1, "Hz", "Character"),
    ("breath", "Breathiness", 0, 1, 0.01, "", "Character"),
    ("ring_mix", "Metallic (ring mod)", 0, 1, 0.01, "", "Character"),
    ("ring_freq", "Ring frequency", 10, 1000, 1, "Hz", "Character"),
    ("drive", "Distortion", 0, 1, 0.01, "", "Character"),
    ("crush_bits", "Bit depth", 2, 16, 1, "bit", "Character"),
    ("crush_down", "Downsample", 1, 24, 1, "x", "Character"),
    ("chorus_mix", "Chorus / flanger", 0, 1, 0.01, "", "Space"),
    ("chorus_rate", "Mod rate", 0.05, 8, 0.05, "Hz", "Space"),
    ("chorus_delay", "Mod delay", 1, 30, 0.5, "ms", "Space"),
    ("chorus_depth", "Mod depth", 0, 8, 0.1, "ms", "Space"),
    ("echo_mix", "Echo", 0, 1, 0.01, "", "Space"),
    ("echo_time", "Echo time", 40, 1000, 5, "ms", "Space"),
    ("echo_fb", "Echo feedback", 0, 0.9, 0.01, "", "Space"),
    ("reverb_mix", "Reverb", 0, 1, 0.01, "", "Space"),
    ("reverb_size", "Room size", 0, 1, 0.01, "", "Space"),
    ("low_cut", "Low cut", 20, 1000, 5, "Hz", "Tone"),
    ("high_cut", "High cut", 1000, 20000, 100, "Hz", "Tone"),
    ("bass", "Bass", -12, 12, 0.5, "dB", "Tone"),
    ("treble", "Treble", -12, 12, 0.5, "dB", "Tone"),
    ("gain", "Output gain", -12, 12, 0.5, "dB", "Tone"),
]

DEFAULTS = {
    "pitch": 0, "formant": 0, "tremble": 0, "robot_mix": 0, "robot_pitch": 120,
    "breath": 0, "ring_mix": 0, "ring_freq": 50, "drive": 0, "crush_bits": 16,
    "crush_down": 1, "chorus_mix": 0, "chorus_rate": 0.8, "chorus_delay": 12,
    "chorus_depth": 2, "echo_mix": 0, "echo_time": 250, "echo_fb": 0.35,
    "reverb_mix": 0, "reverb_size": 0.6, "low_cut": 20, "high_cut": 20000,
    "bass": 0, "treble": 0, "gain": 0,
}


def make(name, color, **kw):
    p = dict(DEFAULTS)
    p.update(kw)
    if "formant" not in kw:  # formants follow pitch unless told otherwise
        p["formant"] = p["pitch"]
    return {"name": name, "color": color, "params": p, "builtin": True}


BUILTIN = [
    make("Clean", "#6b7280"),
    make("Girl", "#f472b6", pitch=7, formant=3.5, low_cut=170, treble=2.5, bass=-3, gain=2),
    make("Woman", "#ec4899", pitch=5, formant=2.5, low_cut=130, treble=1.5, bass=-2, gain=1.5),
    make("Old Man", "#a8a29e", pitch=-2, formant=-1, tremble=0.6, breath=0.22,
         low_cut=160, high_cut=6500, drive=0.04, treble=-1, gain=6),
    make("Robot", "#10b981", robot_mix=1.0, robot_pitch=115, chorus_mix=0.3,
         chorus_rate=0.1, chorus_delay=2.5, chorus_depth=0.05, treble=2, low_cut=90),
    make("Mega Echo", "#6366f1", echo_mix=0.85, echo_time=320, echo_fb=0.78,
         reverb_mix=0.25, reverb_size=0.85, gain=-0.5),
    make("Deep Voice", "#3b82f6", pitch=-4, bass=3, low_cut=60),
    make("Chipmunk", "#f59e0b", pitch=9, low_cut=150, gain=2),
    make("Child", "#fbbf24", pitch=6, low_cut=180, treble=1.5, gain=3),
    make("Giant", "#1d4ed8", pitch=-9, bass=4, reverb_mix=0.25, reverb_size=0.85),
    make("Demon", "#dc2626", pitch=-8, drive=0.25, bass=4, reverb_mix=0.3,
         reverb_size=0.75, chorus_mix=0.3, chorus_rate=0.3, chorus_delay=8, gain=-1),
    make("Villain", "#7f1d1d", pitch=-5, drive=0.08, bass=3, high_cut=7000,
         reverb_mix=0.15, reverb_size=0.4),
    make("Cyborg", "#14b8a6", ring_mix=0.85, ring_freq=55, chorus_mix=0.5,
         chorus_rate=0.1, chorus_delay=6, chorus_depth=0.1, drive=0.05, gain=4.5),
    make("Android", "#06b6d4", pitch=-1, ring_mix=0.6, ring_freq=110, crush_bits=10,
         treble=3, low_cut=150, gain=7),
    make("Alien", "#22c55e", pitch=3, ring_mix=0.35, ring_freq=380, chorus_mix=0.7,
         chorus_rate=3.5, chorus_delay=4, chorus_depth=2.5, gain=6),
    make("Radio", "#a16207", drive=0.3, low_cut=400, high_cut=3200, crush_bits=12, gain=6.5),
    make("Telephone", "#78716c", low_cut=450, high_cut=3000, crush_down=3, drive=0.1, gain=8),
    make("Megaphone", "#ea580c", drive=0.55, low_cut=600, high_cut=4200, treble=4,
         reverb_mix=0.1, reverb_size=0.3, gain=7.5),
    make("8-Bit", "#a855f7", pitch=5, crush_bits=4, crush_down=8, low_cut=120),
    make("Cave", "#57534e", reverb_mix=0.6, reverb_size=0.92, echo_mix=0.25,
         echo_time=320, echo_fb=0.3),
    make("Stadium Announcer", "#0ea5e9", bass=2, treble=2, echo_mix=0.35,
         echo_time=180, echo_fb=0.4, reverb_mix=0.3, reverb_size=0.8),
    make("Ghost", "#cbd5e1", pitch=2, chorus_mix=0.8, chorus_rate=0.25,
         chorus_delay=3, chorus_depth=2.5, reverb_mix=0.55, reverb_size=0.9,
         low_cut=250, gain=7),
    make("Underwater", "#0369a1", pitch=-2, high_cut=1200, chorus_mix=0.7,
         chorus_rate=5, chorus_delay=6, chorus_depth=3, reverb_mix=0.2, gain=4),
]
