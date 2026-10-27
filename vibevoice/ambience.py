"""Background layers: procedurally synthesised ambience mixed under the voice.

Nothing is loaded from disk. When a layer is switched on, its sounds are
rendered once in a background thread (a looping bed plus banks of one-shot
events); playback then only mixes pre-rendered arrays, so it costs almost
no CPU. Events fire at random (Poisson) times, so a layer never repeats.
"""
import math
import threading

import numpy as np

from .dsp import F32, Reverb

SR = 48000
TARGET_RMS = 10 ** (-24 / 20)  # loudness of every layer at volume 1


# --------------------------------------------------------------------------
# synthesis helpers (offline, vectorised)
# --------------------------------------------------------------------------
def _n(dur):
    return max(1, int(dur * SR))


def _t(n):
    return np.arange(n) / SR


def curve(points, n):
    """Piecewise-linear curve over n samples; points = [(0..1, value), ...] or a number."""
    if np.isscalar(points):
        return np.full(n, float(points))
    xs, ys = zip(*points)
    return np.interp(np.linspace(0, 1, n), xs, ys)


def _shape(x, lo, hi, tilt=0.0):
    """Band-limit and tilt (dB/octave) a signal in the frequency domain."""
    n = len(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = 1.0
    g = (f / 1000.0) ** (tilt / 6.02) / np.sqrt(1 + (lo / f) ** 4) / np.sqrt(1 + (f / hi) ** 4)
    return np.fft.irfft(np.fft.rfft(x) * g, n)


def noise(n, rng, lo=20, hi=20000, tilt=0.0):
    """Shaped Gaussian noise with RMS 1."""
    y = _shape(rng.standard_normal(n), lo, hi, tilt)
    return y / (np.sqrt(np.mean(y ** 2)) + 1e-12)


def wander(n, rate, rng):
    """Smooth random modulation in 0..1, changing about `rate` times a second."""
    k = max(2, int(n / SR * rate) + 2)
    pts = rng.random(k)
    x = np.linspace(0, k - 1.001, n)
    i = x.astype(int)
    w = (1 - np.cos(np.pi * (x - i))) / 2
    return pts[i] * (1 - w) + pts[i + 1] * w


def ar(n, attack, release, power=1.5):
    t = _t(n)
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    r = np.clip((n / SR - t) / max(release, 1e-4), 0, 1)
    return a * r ** power


def norm(x, peak=1.0):
    m = np.max(np.abs(x))
    return x * (peak / m) if m > 0 else x


def sil(dur):
    return np.zeros(_n(dur))


def cat(*parts):
    return np.concatenate([np.atleast_1d(p) for p in parts])


def _additive(f, amp_of):
    """Sum of harmonics of the (time-varying) f0 `f`; amp_of(k, fk) -> amplitude per control frame."""
    n = len(f)
    ph = 2 * np.pi * np.cumsum(f) / SR
    hop = 128
    ci = np.arange(0, n, hop)
    idx = np.arange(n)
    fc = f[ci]
    out = np.zeros(n)
    for k in range(1, int(min(90, 18000 / max(30.0, f.min()))) + 1):
        a = amp_of(k, k * fc) * (k * fc < 18000)
        if a.max() < 1e-3:  # inaudible
            continue
        out += np.interp(idx, ci, a) * np.sin(k * ph)
    return out


def voice(dur, f0, formants, rng, breath=0.0, jitter=0.0, vibrato=(0.0, 0.0), tilt=1.0,
          drive=0.0, attack=0.02, release=0.1):
    """Source-filter animal/vocal synth: harmonics of f0 shaped by resonances.

    formants: [(freq or curve points, bandwidth Hz, gain), ...]
    """
    n = _n(dur)
    f = curve(f0, n)
    if vibrato[0]:
        f = f * (1 + vibrato[1] * np.sin(2 * np.pi * vibrato[0] * _t(n)))
    if jitter:
        f = f * (1 + jitter * (2 * wander(n, 40, rng) - 1))
    ci = np.arange(0, n, 128)
    res = [(curve(F, n)[ci], bw / 2, g) for F, bw, g in formants]

    def amp(k, fk):
        a = sum(g / (1 + ((fk - F) / hb) ** 2) for F, hb, g in res)
        return a / k ** (0.5 * tilt)

    y = norm(_additive(f, amp))
    if breath:
        lo = min(float(np.min(r[0])) for r in res) * 0.6
        hi = max(float(np.max(r[0])) for r in res) * 1.6
        y = y + breath * noise(n, rng, lo, hi) * 0.35
    if drive:
        y = np.tanh(drive * y)
    return norm(y * ar(n, attack, release))


def tone(dur, f, kind="saw", lp=4000.0, vibrato=(0.0, 0.0)):
    """Band-limited saw/square tone with a gentle low-pass, f = Hz or curve points."""
    n = _n(dur)
    fr = curve(f, n)
    if vibrato[0]:
        fr = fr * (1 + vibrato[1] * np.sin(2 * np.pi * vibrato[0] * _t(n)))
    odd = kind == "square"

    def amp(k, fk):
        if odd and k % 2 == 0:
            return np.zeros_like(fk)
        return (1.0 / k) / (1 + (fk / lp) ** 2)

    return norm(_additive(fr, amp))


def sweep(dur, f, harmonics=(1.0,)):
    """Sine sweep (with optional harmonics) following f = Hz or curve points."""
    n = _n(dur)
    ph = 2 * np.pi * np.cumsum(curve(f, n)) / SR
    return norm(sum(a * np.sin((i + 1) * ph) for i, a in enumerate(harmonics)))


def modal(dur, f, ratios, amps, decays):
    """Struck object: decaying inharmonic partials (bells, hammers, metal)."""
    t = _t(_n(dur))
    return norm(sum(a * np.sin(2 * np.pi * f * r * t) * np.exp(-t / d)
                    for r, a, d in zip(ratios, amps, decays)))


def space(x, wet=0.3, size=0.7, lp=None):
    """Distance: optional low-pass plus a reverb tail."""
    if lp:
        x = _shape(x, 10, lp)
    x = cat(x, sil(0.4 + 1.6 * size))
    rv = Reverb(SR)
    out = np.concatenate([rv.process(x[i:i + 4096].astype(F32), wet, size)
                          for i in range(0, len(x), 4096)])
    return norm(out)


def loop(x, fade=1.0):
    """Make a seamless loop by cross-fading the tail into the head."""
    f = _n(fade)
    body = x[:-f].copy()
    w = np.sin(np.linspace(0, np.pi / 2, f)) ** 2
    body[:f] = x[:f] * w + x[-f:] * (1 - w)
    return body


def impulses(n, rate, rng, heavy=1.5):
    """Sparse random clicks with heavy-tailed sizes (rain drops, fire crackle)."""
    x = np.zeros(n)
    k = rng.poisson(rate * n / SR)
    sizes = np.minimum(rng.pareto(heavy, k), 6.0)  # mostly small, a few big, never extreme
    x[rng.integers(0, n, k)] = sizes * rng.choice((-1, 1), k)
    return x


# --------------------------------------------------------------------------
# sounds
# --------------------------------------------------------------------------
def U(rng, a, b):
    return float(rng.uniform(a, b))


def monkey(rng):
    calls, n = [], int(rng.integers(4, 8))
    b = U(rng, 330, 430)
    for i in range(n):
        fr = i / (n - 1)
        f = b * (1 + 0.8 * fr)
        calls += [voice(U(rng, 0.13, 0.2), [(0, f), (1, f * 1.12)],
                        [(350 + 550 * fr, 160, 1), (850 + 600 * fr, 220, 0.6), (2800, 400, 0.2)],
                        rng, breath=0.15, jitter=0.03, attack=0.015, release=0.07),
                  sil(U(rng, 0.05, 0.1))]
    return space(cat(*calls), 0.2, 0.5)


def elephant(rng):
    b = U(rng, 360, 480)
    y = voice(U(rng, 1.0, 1.4), [(0, b * 0.8), (0.2, b * 1.15), (0.7, b), (1, b * 0.7)],
              [(900, 500, 1), (1800, 600, 0.8), (3200, 900, 0.6)], rng, breath=0.35,
              jitter=0.04, vibrato=(7, 0.025), tilt=0.4, drive=2.5, attack=0.08, release=0.4)
    return space(y, 0.25, 0.6)


def lion(rng):
    s = U(rng, 0.85, 1.15)
    y = voice(U(rng, 1.4, 1.8), [(0, 170 * s), (0.15, 210 * s), (0.6, 140 * s), (1, 85 * s)],
              [(650, 300, 1), (1100, 350, 0.7), (2500, 600, 0.3)], rng, breath=0.6,
              jitter=0.12, drive=2.0, attack=0.12, release=0.8)
    y = y * (0.55 + 0.45 * wander(len(y), 35, rng))  # growl
    return space(y, 0.2, 0.6, lp=3500)


def duck(rng):
    parts = []
    for _ in range(int(rng.integers(2, 5))):
        s = U(rng, 0.9, 1.1)
        parts += [voice(U(rng, 0.16, 0.22), [(0, 420 * s), (1, 330 * s)],
                        [(1100, 250, 1), (2300, 300, 0.9), (3300, 400, 0.5)], rng,
                        breath=0.25, jitter=0.05, tilt=0.3, drive=1.5, attack=0.01,
                        release=0.12), sil(U(rng, 0.08, 0.15))]
    return space(cat(*parts), 0.15, 0.4)


def sheep(rng):
    b = U(rng, 230, 300)
    y = voice(U(rng, 0.7, 0.95), [(0, b), (0.3, b * 1.05), (1, b * 0.92)],
              [(850, 200, 1), (1500, 250, 0.7), (2700, 400, 0.3)], rng, breath=0.15,
              vibrato=(7.5, 0.06), attack=0.03, release=0.25)
    y = y * (0.65 + 0.35 * np.sin(2 * np.pi * 7.5 * _t(len(y))))  # bleat
    return space(y, 0.2, 0.5)


def cow(rng):
    s = U(rng, 0.85, 1.15)
    y = voice(U(rng, 1.2, 1.6), [(0, 120 * s), (0.25, 150 * s), (0.8, 135 * s), (1, 100 * s)],
              [([(0, 250), (0.3, 450), (1, 400)], 120, 1),
               ([(0, 900), (0.3, 800), (1, 750)], 200, 0.5), (2400, 400, 0.15)],
              rng, breath=0.1, jitter=0.02, attack=0.15, release=0.5)
    return space(y, 0.25, 0.6, lp=2500)


def parrot(rng):
    parts = []
    for _ in range(int(rng.integers(1, 3))):
        s = U(rng, 0.85, 1.15)
        parts += [voice(U(rng, 0.3, 0.45), [(0, 850 * s), (0.3, 1000 * s), (1, 650 * s)],
                        [(1800, 500, 1), (3200, 700, 0.8)], rng, breath=0.5, jitter=0.15,
                        drive=3, attack=0.01, release=0.15), sil(U(rng, 0.1, 0.2))]
    return space(cat(*parts), 0.15, 0.4)


def chirps(rng, lo=2500, hi=6000):
    parts = []
    for _ in range(int(rng.integers(3, 8))):
        a, b = U(rng, lo, hi), U(rng, lo, hi)
        d = U(rng, 0.05, 0.09)
        c = sweep(d, [(0, a), (1, b)], (1.0, 0.15))
        parts += [c * np.sin(np.pi * np.linspace(0, 1, len(c))), sil(U(rng, 0.03, 0.08))]
    return space(cat(*parts), 0.25, 0.5)


def dog(rng):
    parts = []
    for _ in range(int(rng.integers(2, 4))):
        s = U(rng, 0.85, 1.15)
        parts += [voice(U(rng, 0.15, 0.2), [(0, 550 * s), (0.2, 600 * s), (1, 380 * s)],
                        [(700, 250, 1), (1300, 300, 0.8), (2600, 500, 0.4)], rng,
                        breath=0.45, jitter=0.06, drive=2.0, attack=0.005, release=0.1),
                  sil(U(rng, 0.12, 0.25))]
    return space(cat(*parts), 0.3, 0.6)


def wave(rng):
    n = _n(U(rng, 7, 10))
    p = U(rng, 0.35, 0.45)
    env = curve([(0, 0), (p * 0.7, 0.3), (p, 1), (p + 0.1, 0.75), (1, 0)], n) ** 1.5
    body = noise(n, rng, 40, 900, -3) * env
    wash = noise(n, rng, 300, 3000, -2) * env ** 2 * 0.5
    fizz = noise(n, rng, 1500, 14000, -1.5) * env ** 3 * 0.6
    return norm(body + wash + fizz)


def gull(rng):
    parts = []
    b = U(rng, 850, 1150)
    for _ in range(int(rng.integers(2, 5))):
        parts += [voice(U(rng, 0.3, 0.45), [(0, b * 1.15), (0.25, b * 1.25), (1, b * 0.75)],
                        [(1900, 500, 1), (3300, 700, 0.6)], rng, breath=0.2, jitter=0.03,
                        drive=1.2, attack=0.02, release=0.2), sil(U(rng, 0.1, 0.2))]
    return space(cat(*parts), 0.35, 0.7)


def ship_horn(rng):
    f = U(rng, 85, 140)
    d = U(rng, 2.2, 3.5)
    bend = [(0, f * 0.97), (0.08, f), (1, f)]
    y = tone(d, bend, "saw", 1200, (4, 0.004)) + 0.7 * tone(d, [(a, v * 1.5) for a, v in bend],
                                                           "saw", 1200, (4.3, 0.004))
    y = norm(y) * ar(len(y), 0.25, 0.5)
    y = cat(y, sil(1.0))
    lag = _n(0.9)
    y[lag:] += 0.35 * y[:-lag]  # harbour echo
    return space(y, 0.5, 0.95)


def bell(rng):
    f = U(rng, 450, 600)
    hit = modal(2.5, f, (1, 2.0, 2.76, 4.07, 5.4), (1, .5, .6, .3, .2), (2.5, 1.8, 1.2, .8, .6))
    parts = []
    for _ in range(int(rng.integers(2, 4))):
        parts += [hit * U(rng, 0.6, 1.0)]
    out = np.zeros(sum(len(p) for p in parts))
    pos = 0
    for p in parts:
        out[pos:pos + len(p)] += p
        pos += _n(U(rng, 0.8, 1.4))
    return space(out[:pos + len(parts[-1])], 0.3, 0.7)


def thunder(rng):
    n = _n(U(rng, 5, 8))
    near = rng.random() < 0.5
    rumble = noise(n, rng, 20, 600 if near else 250, -4)
    t = _t(n)
    env = np.zeros(n)
    for _ in range(int(rng.integers(4, 9))):
        t0 = U(rng, 0, 1.8)
        env += U(rng, 0.3, 1) * np.where(t >= t0, np.exp(-(t - t0) / U(rng, 0.6, 1.8)), 0)
    y = rumble * env * (0.6 + 0.4 * wander(n, 3, rng))
    if near:
        crack = noise(_n(0.5), rng, 600, 9000, -3) * np.exp(-_t(_n(0.5)) / 0.12)
        y[:len(crack)] += 0.8 * crack * np.sqrt(np.mean(y ** 2)) / 0.3  # sharp edge, rumble dominates
    y = norm(y)
    return y * ar(n, 0.005, 1.5)


def car(rng):
    d = U(rng, 3, 4.5)
    n = _n(d)
    t = _t(n)
    t0 = d * U(rng, 0.4, 0.6)
    f = U(rng, 70, 110)
    doppler = f * (1 - 0.06 * np.tanh((t - t0) / 0.4))
    engine = tone(d, list(zip(np.linspace(0, 1, 64), doppler[np.linspace(0, n - 1, 64).astype(int)])),
                  "saw", 900)
    tyres = noise(n, rng, 300, 5000, -2)
    env = np.exp(-((t - t0) / (d * 0.22)) ** 2)
    return norm((engine + 0.6 * tyres) * env)


def car_horn(rng):
    f = U(rng, 380, 450)
    parts = []
    for _ in range(int(rng.integers(1, 4))):
        d = U(rng, 0.2, 0.7)
        h = tone(d, f, "square", 3000) + tone(d, f * 1.19, "square", 3000)
        parts += [norm(h) * ar(len(h), 0.01, 0.05), sil(U(rng, 0.08, 0.18))]
    return space(cat(*parts), 0.2, 0.5)


def siren(rng):
    d = U(rng, 6, 9)
    t = np.linspace(0, d, 200)
    if rng.random() < 0.5:
        f = 900 + 350 * np.sin(2 * np.pi * 0.3 * t)  # wail
    else:
        f = 1000 + 300 * np.sin(2 * np.pi * 3 * t)  # yelp
    y = tone(d, list(zip(t / d, f)), "square", 4000)
    return space(y * ar(len(y), 2.0, 2.0, 1.0), 0.3, 0.7)


def frog(rng):
    parts = []
    f = U(rng, 110, 220)
    for _ in range(int(rng.integers(2, 4))):
        c = voice(U(rng, 0.08, 0.12), f, [(600, 250, 1), (1600, 400, 0.6)], rng, breath=0.2,
                  jitter=0.05, attack=0.005, release=0.04)
        c = c * (0.5 + 0.5 * np.sin(2 * np.pi * U(rng, 30, 50) * _t(len(c)))) ** 2
        parts += [c, sil(0.05)]
    return space(cat(*parts, sil(0.15), *parts), 0.25, 0.5)


def tropical_bird(rng):
    parts = []
    for _ in range(int(rng.integers(1, 4))):
        d = U(rng, 0.2, 0.6)
        a, b = U(rng, 1500, 4000), U(rng, 1500, 4000)
        n = _n(d)
        f = curve([(0, a), (1, b)], n) * (1 + U(rng, 0.05, 0.1) *
                                          np.sin(2 * np.pi * U(rng, 12, 25) * _t(n)))
        c = sweep(d, list(zip(np.linspace(0, 1, 128), f[np.linspace(0, n - 1, 128).astype(int)])),
                  (1.0, 0.2))
        parts += [c * ar(n, 0.02, 0.1), sil(U(rng, 0.05, 0.15))]
    return space(cat(*parts), 0.3, 0.6)


def fire_pop(rng):
    n = _n(0.25)
    y = noise(n, rng, 500, 4000) * np.exp(-_t(n) / 0.02)
    y += 0.4 * _shape(impulses(n, 40, rng), 1500, 9000)
    return norm(y)


def crickets(rng):
    pulse = sweep(0.015, U(rng, 4300, 4900))
    pulse = pulse * np.sin(np.pi * np.linspace(0, 1, len(pulse)))
    group = cat(*[cat(pulse, sil(0.015)) for _ in range(int(rng.integers(3, 6)))])
    return norm(cat(*[cat(group, sil(U(rng, 0.2, 0.35))) for _ in range(int(rng.integers(3, 7)))]))


def owl(rng):
    f = U(rng, 340, 400)
    def hoo(d):
        return voice(d, [(0, f), (1, f * 0.95)], [(f, 150, 1), (800, 300, 0.1)], rng,
                     breath=0.05, attack=0.05, release=0.15)
    return space(cat(hoo(0.35), sil(0.25), hoo(0.18), sil(0.08), hoo(0.4)), 0.4, 0.8)


def jackhammer(rng):
    n = _n(U(rng, 1.5, 3))
    hit_n = _n(0.04)
    th = _t(hit_n)
    hit = (noise(hit_n, rng, 500, 6000) * np.exp(-th / 0.006)
           + modal(0.04, U(rng, 2000, 2300), (1, 1.6), (0.6, 0.4), (0.02, 0.015))
           + 1.5 * np.sin(2 * np.pi * 90 * th) * np.exp(-th / 0.025))
    y = np.zeros(n + hit_n)
    for pos in np.arange(0, n, SR / U(rng, 18, 24)).astype(int):
        y[pos:pos + hit_n] += hit * U(rng, 0.8, 1.0)
    return space(norm(y * ar(len(y), 0.05, 0.2)), 0.15, 0.4)


def hammer(rng):
    f = U(rng, 600, 1100)
    knock = modal(0.3, f, (1, 2.3, 3.9), (1, 0.5, 0.3), (0.06, 0.04, 0.03))
    knock[:_n(0.004)] += noise(_n(0.004), rng, 1000, 10000) * 0.8
    out = np.zeros(_n(4))
    pos = 0
    for _ in range(int(rng.integers(3, 7))):
        out[pos:pos + len(knock)] += knock * U(rng, 0.7, 1.0)
        pos += _n(U(rng, 0.35, 0.5))
    return space(out[:pos + len(knock)], 0.2, 0.5)


def drill(rng):
    d = U(rng, 1.5, 3)
    f = U(rng, 180, 260)
    y = tone(d, [(0, f * 0.5), (0.15, f), (0.85, f), (1, f * 0.4)], "saw", 5000)
    y = y + 0.3 * noise(len(y), rng, 2000, 8000)
    y = y * (0.85 + 0.15 * np.sin(2 * np.pi * 25 * _t(len(y))))
    return space(norm(y) * ar(len(y), 0.1, 0.3), 0.15, 0.4)


def beeper(rng):
    f = U(rng, 1000, 1100)
    beep = sweep(0.45, f, (1.0, 0, 0.15))
    beep = beep * ar(len(beep), 0.005, 0.01)
    return space(cat(*[cat(beep, sil(0.45)) for _ in range(int(rng.integers(5, 9)))]), 0.25, 0.6)


def laser(rng):
    parts = []
    for _ in range(int(rng.integers(1, 4))):
        d = U(rng, 0.12, 0.25)
        n = _n(d)
        a, b = U(rng, 1800, 3500), U(rng, 200, 500)
        f = a * (b / a) ** np.linspace(0, 1, n)
        c = tone(d, list(zip(np.linspace(0, 1, 64), f[np.linspace(0, n - 1, 64).astype(int)])),
                 "square", 6000)
        parts += [c * np.exp(-_t(n) / (d * 0.5)), sil(U(rng, 0.07, 0.12))]
    return space(cat(*parts), 0.15, 0.4)


def explosion(rng):
    n = _n(U(rng, 2, 3))
    t = _t(n)
    body = noise(n, rng, 20, 3000, -3) * np.exp(-t / 0.5)
    boom = np.sin(2 * np.pi * np.cumsum(70 * (0.5 ** (t / 1.0))) / SR) * np.exp(-t / 0.8) * 2
    debris = noise(n, rng, 1000, 6000) * np.exp(-t / 0.3) * wander(n, 25, rng) * 0.7
    y = np.tanh(2 * norm(body + boom + debris))
    return norm(y * ar(n, 0.005, 0.8))


def alarm(rng):
    whoop = tone(0.6, [(0, 400), (1, 1100)], "square", 3000)
    whoop = whoop * ar(len(whoop), 0.01, 0.05)
    return space(cat(*[cat(whoop, sil(0.15)) for _ in range(int(rng.integers(3, 5)))]), 0.3, 0.6)


def robot_chatter(rng):
    parts = []
    for _ in range(int(rng.integers(4, 9))):
        d = U(rng, 0.04, 0.09)
        a, b = U(rng, 800, 2400), U(rng, 800, 2400)
        c = sweep(d, [(0, a), (1, b)], (1.0, 0.3))
        parts += [c * ar(len(c), 0.003, 0.01), sil(U(rng, 0.02, 0.06))]
    return space(cat(*parts), 0.2, 0.4)


# --------------------------------------------------------------------------
# layers
# --------------------------------------------------------------------------
def _bed(dur, fn):
    return loop(fn(_n(dur + 1.0)))


def _zoo(rng):
    beds = [(_bed(12, lambda n: noise(n, rng, 200, 1500, -3) * (0.6 + 0.4 * wander(n, 0.2, rng))), 0.3)]
    return beds, [
        (monkey, 3, 10, (0.25, 0.5)), (elephant, 2, 5, (0.2, 0.4)), (lion, 2, 4, (0.25, 0.45)),
        (duck, 3, 8, (0.15, 0.3)), (sheep, 3, 6, (0.2, 0.35)), (cow, 2, 4, (0.25, 0.4)),
        (parrot, 3, 6, (0.12, 0.25)), (chirps, 4, 12, (0.08, 0.18)), (dog, 3, 5, (0.2, 0.35)),
    ]


def _sea(rng):
    beds = [(_bed(14, lambda n: noise(n, rng, 30, 500, -4) * (0.7 + 0.3 * wander(n, 0.15, rng))), 1.0),
            (_bed(14, lambda n: noise(n, rng, 2000, 12000, -2) * (0.5 + 0.5 * wander(n, 0.3, rng))), 0.25)]
    return beds, [(wave, 4, 9, (0.35, 0.6)), (gull, 4, 7, (0.12, 0.25)),
                  (ship_horn, 3, 2, (0.3, 0.5)), (bell, 2, 2, (0.1, 0.18))]


def _storm(rng):
    def rain(n):
        drops = _shape(impulses(n, 150, rng), 2000, 9000)
        drops = norm(drops) * 2.5  # by peak: sparse clicks would spike if scaled by RMS
        return noise(n, rng, 400, 16000, -1.5) * (0.85 + 0.15 * wander(n, 0.5, rng)) + drops

    def wind(n):
        return sum(noise(n, rng, lo, hi, -3) * wander(n, r, rng) ** 2
                   for lo, hi, r in ((150, 500, 0.1), (500, 1200, 0.2), (1200, 3000, 0.3)))
    beds = [(_bed(12, rain), 1.0), (_bed(16, wind), 0.6)]
    return beds, [(thunder, 4, 4, (0.5, 0.8))]


def _city(rng):
    beds = [(_bed(12, lambda n: noise(n, rng, 25, 400, -4) * (0.7 + 0.3 * wander(n, 0.2, rng))), 1.0),
            (_bed(12, lambda n: noise(n, rng, 100, 3000, -3)), 0.3)]
    return beds, [(car, 4, 10, (0.2, 0.35)), (car_horn, 4, 5, (0.2, 0.4)),
                  (siren, 3, 1.5, (0.15, 0.3)), (dog, 2, 2, (0.1, 0.2))]


def _jungle(rng):
    def insects(n):
        t = _t(n)
        a = noise(n, rng, 4000, 7000) * (0.5 + 0.5 * np.sin(2 * np.pi * 45 * t)) ** 4
        b = noise(n, rng, 5500, 8000) * wander(n, 0.1, rng) ** 2
        return a + 0.7 * b + 0.6 * noise(n, rng, 80, 600, -3)
    beds = [(_bed(12, insects), 1.0)]
    return beds, [(frog, 4, 14, (0.15, 0.3)), (tropical_bird, 5, 20, (0.1, 0.25)),
                  (monkey, 3, 5, (0.2, 0.35)), (parrot, 2, 4, (0.12, 0.22))]


def _campfire(rng):
    def fire(n):
        roar = noise(n, rng, 60, 900, -3) * (0.6 + 0.4 * wander(n, 2, rng))
        crackle = _shape(impulses(n, 25, rng), 1500, 9000)
        return roar + norm(crackle) * 4.0  # by peak: sparse clicks would spike if scaled by RMS
    return [(_bed(12, fire), 1.0)], [(fire_pop, 5, 8, (0.15, 0.3)), (crickets, 3, 6, (0.05, 0.1)),
                                     (owl, 2, 1.5, (0.1, 0.2))]


def _construction(rng):
    def site(n):
        hum = tone(n / SR, 60, "saw", 600) * (0.95 + 0.05 * wander(n, 0.5, rng))
        return hum + noise(n, rng, 50, 400, -3)
    return [(_bed(10, site), 1.0)], [
        (jackhammer, 3, 4, (0.2, 0.35)), (hammer, 4, 6, (0.2, 0.35)),
        (drill, 3, 3, (0.15, 0.3)), (beeper, 2, 2, (0.1, 0.18))]


def _space(rng):
    def engine(n):
        d = n / SR
        return (tone(d, 55, "saw", 400) + tone(d, 55.6, "saw", 400)
                + 0.8 * noise(n, rng, 30, 300, -4)) * (0.8 + 0.2 * wander(n, 0.3, rng))
    return [(_bed(10, engine), 1.0)], [
        (laser, 5, 25, (0.15, 0.3)), (explosion, 3, 4, (0.4, 0.7)),
        (alarm, 2, 1.5, (0.12, 0.25)), (robot_chatter, 4, 5, (0.1, 0.2))]


# id, name, colour, description, builder
LAYERS = [
    ("zoo", "Zoo", "#22c55e", "Monkeys, elephants, lions, ducks, sheep, cows, parrots and dogs, all at once", _zoo),
    ("sea", "Sea", "#0ea5e9", "Crashing waves, seagulls, boat horns and a bell buoy", _sea),
    ("storm", "Storm", "#64748b", "Heavy rain, gusting wind and rolling thunder", _storm),
    ("city", "City Traffic", "#f59e0b", "Passing cars, honking horns, sirens and a barking dog", _city),
    ("jungle", "Jungle", "#16a34a", "Frogs, buzzing insects, tropical birds and monkeys", _jungle),
    ("campfire", "Campfire", "#ea580c", "Crackling fire, crickets and an owl", _campfire),
    ("construction", "Construction", "#eab308", "Jackhammers, hammering, drills and reversing beepers", _construction),
    ("space", "Space Battle", "#a855f7", "Laser blasts, explosions, alarms and robot chatter", _space),
]
LAYER_IDS = [lay[0] for lay in LAYERS]


# --------------------------------------------------------------------------
# banks (rendered once per layer, in the background)
# --------------------------------------------------------------------------
class Bank:
    def __init__(self, beds, events):
        self.beds = [(b.astype(F32), g) for b, g in beds]
        # (variants, events per minute, (gain lo, gain hi))
        self.events = [([v.astype(F32) for v in variants], rate, gains)
                       for variants, rate, gains in events]
        self.norm = 1.0


BANKS = {}
_building = set()
_lock = threading.Lock()


def build(layer_id, seed=None):
    rng = np.random.default_rng(seed)
    builder = next(lay[4] for lay in LAYERS if lay[0] == layer_id)
    beds, events = builder(rng)
    beds = [(b / (np.sqrt(np.mean(b ** 2)) + 1e-12) * g * 0.05, 1.0) for b, g in beds]
    bank = Bank(beds, [([fn(rng) for _ in range(k)], rate, gains) for fn, k, rate, gains in events])
    # calibrate: every layer plays at the same loudness at volume 1
    probe = LayerPlayer(SR, seed=1)
    probe.state_for("probe", bank)
    sim = np.concatenate([probe.render_layer("probe", bank, 2048) for _ in range(int(40 * SR / 2048))])
    bank.norm = TARGET_RMS / (float(np.sqrt(np.mean(sim ** 2))) + 1e-9)
    return bank


def prepare(layer_id, on_ready=None):
    """Render a layer's sounds in a background thread (no-op if ready or building)."""
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
            self._fire(st, bank.events[int(self.rng.integers(len(bank.events)))], 0)
        return st

    def _fire(self, st, ev, offset):
        variants, _rate, (glo, ghi) = ev
        clip = variants[int(self.rng.integers(len(variants)))]
        st["voices"].append([clip, -offset, float(self.rng.uniform(glo, ghi))])

    def render_layer(self, lid, bank, n):
        st = self.state_for(lid, bank)
        out = np.zeros(n, F32)
        for j, (bed, g) in enumerate(bank.beds):
            p, L = st["pos"][j], len(bed)
            if p + n <= L:
                out += bed[p:p + n] * g
            else:
                k = L - p
                out[:k] += bed[p:] * g
                out[k:] += bed[:n - k] * g
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
                out[start:start + m] += clip[src:src + m] * g
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
            y = self.render_layer(lid, bank, n) * F32(vol * bank.norm)
            out = y if out is None else out + y
        return out
