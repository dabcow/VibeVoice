"""Real-time voice effects.

Every effect works on mono float32 blocks and keeps its own state, so the
chain can be fed arbitrary block sizes. All per-sample work is vectorised
with numpy / scipy; there are no Python-level per-sample loops.
"""
import math

import numpy as np
from scipy.signal import lfilter, sosfilt

F32 = np.float32


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
class _Ring:
    """Circular sample buffer with block read/write (wrap via two slices)."""

    def __init__(self, size):
        self.buf = np.zeros(size, F32)
        self.size = size
        self.w = 0

    def read_behind(self, delay, n):
        start = (self.w - delay) % self.size
        end = start + n
        if end <= self.size:
            return self.buf[start:end].copy()
        return np.concatenate((self.buf[start:], self.buf[: end - self.size]))

    def write(self, data):
        n = len(data)
        end = self.w + n
        if end <= self.size:
            self.buf[self.w:end] = data
        else:
            k = self.size - self.w
            self.buf[self.w:] = data[:k]
            self.buf[: n - k] = data[k:]
        self.w = end % self.size

    def clear(self):
        self.buf[:] = 0


class FeedbackDelay:
    """y = delayed(store); store = x + fb * lowpass(y).

    Works for delays shorter than the block by processing in sub-chunks of
    at most `delay` samples (each chunk then only depends on the past).
    """

    def __init__(self, max_delay):
        self.ring = _Ring(max_delay + 1)
        self.lp_zi = np.zeros(1, F32)

    def process(self, x, delay, fb, damp=0.0):
        delay = max(1, min(int(delay), self.ring.size - 1))
        n = len(x)
        if n <= delay:
            return self._chunk(x, delay, fb, damp)
        out = np.empty(n, F32)
        for s in range(0, n, delay):
            e = min(s + delay, n)
            out[s:e] = self._chunk(x[s:e], delay, fb, damp)
        return out

    def _chunk(self, x, delay, fb, damp):
        d = self.ring.read_behind(delay, len(x))
        if damp > 0.0:
            lp, self.lp_zi = lfilter([1.0 - damp], [1.0, -damp], d, zi=self.lp_zi)
            self.ring.write(x + fb * lp)
        else:
            self.ring.write(x + fb * d)
        return d

    def clear(self):
        self.ring.clear()
        self.lp_zi[:] = 0


# --------------------------------------------------------------------------
# effects
# --------------------------------------------------------------------------
class NoiseGate:
    """Block-RMS gate with hold and smoothed gain ramps."""

    def __init__(self, sr):
        self.sr = sr
        self.g = 1.0
        self.hold = 0
        self.threshold_db = -50.0

    def process(self, x):
        if self.threshold_db <= -80:
            return x
        rms = math.sqrt(float(np.dot(x, x)) / len(x) + 1e-12)
        db = 20 * math.log10(rms + 1e-12)
        if db > self.threshold_db:
            target, self.hold = 1.0, int(0.18 * self.sr)
        elif self.hold > 0:
            target = 1.0
            self.hold -= len(x)
        else:
            target = 0.0
        coef = 0.7 if target > self.g else 0.12
        g_new = self.g + (target - self.g) * coef
        if g_new == self.g == 1.0:
            return x
        ramp = np.linspace(self.g, g_new, len(x), dtype=F32)
        self.g = g_new
        return x * ramp


class PitchShifter:
    """Two-tap delay-line pitch shifter with sin^2 crossfades (low latency)."""

    def __init__(self, sr, window_ms=40):
        self.W = int(sr * window_ms / 1000)
        self.H = self.W + 4
        self.hist = np.zeros(self.H, F32)
        self.phase = 0.0
        self.ratio = 1.0

    def process(self, x):
        n = len(x)
        buf = np.concatenate((self.hist, x))
        self.hist = buf[-self.H:]
        rate = (1.0 - self.ratio) / self.W
        idx = np.arange(n, dtype=np.float64)
        p1 = (self.phase + idx * rate) % 1.0
        self.phase = (self.phase + n * rate) % 1.0
        p2 = (p1 + 0.5) % 1.0
        base = self.H + idx - 1.0
        return (self._tap(buf, base, p1) + self._tap(buf, base, p2)).astype(F32)

    def _tap(self, buf, base, p):
        pos = base - p * self.W
        i = pos.astype(np.int64)
        f = pos - i
        s = buf[i] * (1.0 - f) + buf[i + 1] * f
        w = np.sin(np.pi * p)
        return s * (w * w)


class RingMod:
    def __init__(self, sr):
        self.sr = sr
        self.phase = 0.0

    def process(self, x, freq, mix):
        n = len(x)
        step = 2 * math.pi * freq / self.sr
        carrier = np.sin(self.phase + step * np.arange(n, dtype=np.float64))
        self.phase = (self.phase + step * n) % (2 * math.pi)
        return x * (1.0 - mix + mix * carrier).astype(F32)


class BitCrusher:
    def __init__(self):
        self.offset = 0
        self.last = 0.0

    def process(self, x, bits, down):
        n = len(x)
        y = x
        if down > 1:
            k = np.arange(n)
            idx = np.where((k + self.offset) % down == 0, k, -1)
            np.maximum.accumulate(idx, out=idx)
            y = np.where(idx >= 0, x[np.maximum(idx, 0)], self.last).astype(F32)
            self.offset = (self.offset + n) % down
            self.last = float(y[-1])
        if bits < 16:
            q = float(2 ** (bits - 1))
            y = np.round(y * q) / q
        return y.astype(F32, copy=False)


class Chorus:
    """Modulated delay (chorus / flanger). Two voices, 90 degrees apart."""

    def __init__(self, sr, max_ms=40):
        self.sr = sr
        self.H = int(sr * max_ms / 1000) + 4
        self.hist = np.zeros(self.H, F32)
        self.phase = 0.0

    def process(self, x, mix, rate, delay_ms, depth_ms):
        n = len(x)
        buf = np.concatenate((self.hist, x))
        self.hist = buf[-self.H:]
        step = 2 * math.pi * rate / self.sr
        ph = self.phase + step * np.arange(n, dtype=np.float64)
        self.phase = (self.phase + step * n) % (2 * math.pi)
        base = self.H + np.arange(n, dtype=np.float64)
        d0 = delay_ms * self.sr / 1000
        dd = depth_ms * self.sr / 1000
        wet = np.zeros(n)
        for off in (0.0, math.pi / 2):
            d = np.clip(d0 + dd * np.sin(ph + off), 1.0, self.H - 3)
            pos = base - d
            i = pos.astype(np.int64)
            f = pos - i
            wet += buf[i] * (1.0 - f) + buf[i + 1] * f
        return (x * (1.0 - 0.5 * mix) + wet * (0.5 * mix)).astype(F32)


class Echo:
    def __init__(self, sr, max_ms=1000):
        self.sr = sr
        self.line = FeedbackDelay(int(sr * max_ms / 1000) + 1)

    def process(self, x, mix, time_ms, fb):
        d = self.line.process(x, time_ms * self.sr / 1000, fb, damp=0.35)
        return x + d * mix


class Reverb:
    """Freeverb-style: 8 damped combs in parallel -> 4 allpasses in series."""

    COMBS = (1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617)
    ALLPASS = (556, 441, 341, 225)

    def __init__(self, sr):
        k = sr / 44100
        self.comb_d = [int(c * k) for c in self.COMBS]
        self.ap_d = [int(a * k) for a in self.ALLPASS]
        self.combs = [FeedbackDelay(d + 1) for d in self.comb_d]
        self.aps = [FeedbackDelay(d + 1) for d in self.ap_d]

    def process(self, x, mix, size, damp=0.3):
        fb = 0.7 + 0.28 * size
        inp = x * 0.015
        acc = np.zeros(len(x), F32)
        for line, d in zip(self.combs, self.comb_d):
            acc += line.process(inp, d, fb, damp)
        for line, d in zip(self.aps, self.ap_d):
            acc = line.process(acc, d, 0.5) - acc
        return x * (1.0 - 0.4 * mix) + acc * (3.0 * mix)

    def clear(self):
        for line in self.combs + self.aps:
            line.clear()


class Vibrato:
    """Shaky voice: wet-only modulated delay (pitch wobble) plus light tremolo."""

    def __init__(self, sr, rate=5.5):
        self.sr = sr
        self.rate = rate
        self.H = int(sr * 0.012) + 4
        self.hist = np.zeros(self.H, F32)
        self.phase = 0.0

    def process(self, x, amount):
        n = len(x)
        buf = np.concatenate((self.hist, x))
        self.hist = buf[-self.H:]
        step = 2 * math.pi * self.rate / self.sr
        # slightly irregular wobble: main LFO plus a slow second one
        ph = self.phase + step * np.arange(n, dtype=np.float64)
        self.phase = (self.phase + step * n) % (2 * math.pi * 1000)
        lfo = np.sin(ph) + 0.35 * np.sin(ph * 0.37)
        d = self.sr * (0.005 + 0.0012 * amount * lfo)
        pos = self.H + np.arange(n, dtype=np.float64) - d
        i = pos.astype(np.int64)
        f = pos - i
        y = buf[i] * (1.0 - f) + buf[i + 1] * f
        return (y * (1.0 - 0.18 * amount * (0.5 + 0.5 * np.sin(ph)))).astype(F32)


class SpectralVoice:
    """STFT stage: formant shift, monotone robot (vocoder) and breathiness.

    Spectral envelope comes from a liftered cepstrum. Formants are moved by
    warping that envelope; the robot replaces the voice's own excitation with
    a band-limited pulse train at a fixed pitch; breath swaps in noise.
    Adds one frame (N samples, ~21 ms) of latency, only while active.
    """

    N, HOP, LIFTER = 1024, 256, 90

    def __init__(self, sr):
        self.sr = sr
        n, hop = self.N, self.HOP
        self.win = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n))
        self.ola_scale = hop / (n * 0.5)  # periodic Hann at 75 % overlap sums to 2
        self.frame = np.zeros(n)
        self.ola = np.zeros(n)
        self.inq = np.zeros(0, F32)
        self.outq = np.zeros(hop, F32)  # one hop of priming keeps output whole
        lift = np.zeros(n)
        lift[0] = 1.0
        lift[1:self.LIFTER] = 2.0  # fold negative quefrencies onto positive
        self.lifter = lift
        self.bins = np.arange(n // 2 + 1, dtype=np.float64)
        self.rng = np.random.default_rng()
        # carrier
        self.car_frame = np.zeros(n)
        self.car_phase = 0.0
        self.car_f0 = None
        self.table = None
        # params
        self.alpha = 1.0
        self.robot = 0.0
        self.robot_f0 = 120.0
        self.breath = 0.0

    def configure(self, formant_st, robot, robot_f0, breath):
        self.alpha = 2 ** (formant_st / 12)
        self.robot, self.breath = robot, breath
        self.robot_f0 = robot_f0
        if robot > 0 and self.car_f0 != robot_f0:
            t = 2048
            k = int(min(t // 2 - 1, 0.45 * self.sr / robot_f0))
            spec = np.zeros(t // 2 + 1)
            spec[1:k + 1] = 1.0
            table = np.fft.irfft(spec, t)
            self.table = np.append(table / np.max(np.abs(table)), table[0])
            self.car_f0 = robot_f0

    @property
    def active(self):
        return abs(self.alpha - 1.0) > 1e-3 or self.robot > 0.001 or self.breath > 0.001

    def process(self, x):
        self.inq = np.concatenate((self.inq, x))
        hop = self.HOP
        outs = [self.outq]
        while len(self.inq) >= hop:
            chunk, self.inq = self.inq[:hop], self.inq[hop:]
            outs.append(self._hop(chunk))
        q = np.concatenate(outs)
        out, self.outq = q[:len(x)], q[len(x):]
        return out.astype(F32)

    def _carrier(self, n):
        t = len(self.table) - 1
        step = self.robot_f0 / self.sr * t
        pos = (self.car_phase + step * np.arange(n)) % t
        self.car_phase = (self.car_phase + step * n) % t
        i = pos.astype(np.int64)
        f = pos - i
        return self.table[i] * (1.0 - f) + self.table[i + 1] * f

    def _hop(self, chunk):
        n, hop = self.N, self.HOP
        self.frame[:-hop] = self.frame[hop:]
        self.frame[-hop:] = chunk
        X = np.fft.rfft(self.frame * self.win)
        mag = np.abs(X) + 1e-9
        env = np.fft.rfft(np.fft.irfft(np.log(mag), n) * self.lifter).real
        if abs(self.alpha - 1.0) > 1e-3:
            env_w = np.interp(self.bins / self.alpha, self.bins, env)
            S = X * np.exp(np.clip(env_w - env, -4.0, 4.0))
        else:
            env_w, S = env, X
        energy = float(np.sum(np.abs(S) ** 2))
        if self.robot > 0.001 or self.breath > 0.001:
            shape = np.exp(env_w - env_w.max())
            if self.robot > 0.001:
                self.car_frame[:-hop] = self.car_frame[hop:]
                self.car_frame[-hop:] = self._carrier(hop)
                R = np.fft.rfft(self.car_frame * self.win) * shape
                R *= math.sqrt(energy / (float(np.sum(np.abs(R) ** 2)) + 1e-12))
                S = S * (1.0 - self.robot) + R * self.robot
            if self.breath > 0.001:
                B = shape * np.exp(2j * np.pi * self.rng.random(len(shape)))
                B *= math.sqrt(energy / (float(np.sum(np.abs(B) ** 2)) + 1e-12))
                S = S * (1.0 - self.breath) + B * self.breath
        y = np.fft.irfft(S, n) * self.win * self.ola_scale
        self.ola += y
        out = self.ola[:hop].copy()
        self.ola[:-hop] = self.ola[hop:]
        self.ola[-hop:] = 0.0
        return out


class Limiter:
    def __init__(self):
        self.g = 1.0

    def process(self, x):
        peak = float(np.max(np.abs(x))) if len(x) else 0.0
        target = 0.95 / peak if peak > 0.95 else 1.0
        g_new = target if target < self.g else self.g + (target - self.g) * 0.05
        if g_new == self.g == 1.0:
            return x
        y = x * np.linspace(self.g, g_new, len(x), dtype=F32)
        self.g = g_new
        return np.clip(y, -1.0, 1.0, out=y)


# --------------------------------------------------------------------------
# EQ (RBJ biquads -> one scipy SOS cascade)
# --------------------------------------------------------------------------
def _biquad(kind, f0, sr, gain_db=0.0, q=0.7071):
    w0 = 2 * math.pi * min(f0, sr * 0.45) / sr
    c, s = math.cos(w0), math.sin(w0)
    alpha = s / (2 * q)
    if kind == "hp":
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
        a = [1 + alpha, -2 * c, 1 - alpha]
    elif kind == "lp":
        b = [(1 - c) / 2, 1 - c, (1 - c) / 2]
        a = [1 + alpha, -2 * c, 1 - alpha]
    else:
        A = 10 ** (gain_db / 40)
        sa = 2 * math.sqrt(A) * (s / 2 * math.sqrt(2))
        if kind == "ls":
            b = [A * ((A + 1) - (A - 1) * c + sa), 2 * A * ((A - 1) - (A + 1) * c),
                 A * ((A + 1) - (A - 1) * c - sa)]
            a = [(A + 1) + (A - 1) * c + sa, -2 * ((A - 1) + (A + 1) * c),
                 (A + 1) + (A - 1) * c - sa]
        else:  # high shelf
            b = [A * ((A + 1) + (A - 1) * c + sa), -2 * A * ((A - 1) + (A + 1) * c),
                 A * ((A + 1) + (A - 1) * c - sa)]
            a = [(A + 1) - (A - 1) * c + sa, 2 * ((A - 1) - (A + 1) * c),
                 (A + 1) - (A - 1) * c - sa]
    return [v / a[0] for v in b] + [1.0, a[1] / a[0], a[2] / a[0]]


class EQ:
    def __init__(self, sr):
        self.sr = sr
        self.sos = None
        self.zi = None

    def configure(self, low_cut, high_cut, bass, treble):
        secs = []
        if low_cut > 25:
            secs += [_biquad("hp", low_cut, self.sr)] * 2
        if high_cut < 18000:
            secs += [_biquad("lp", high_cut, self.sr)] * 2
        if abs(bass) > 0.1:
            secs.append(_biquad("ls", 220, self.sr, bass))
        if abs(treble) > 0.1:
            secs.append(_biquad("hs", 3500, self.sr, treble))
        if not secs:
            self.sos = self.zi = None
            return
        sos = np.array(secs, dtype=np.float64)
        if self.zi is None or self.zi.shape[0] != len(secs):
            self.zi = np.zeros((len(secs), 2))
        self.sos = sos

    def process(self, x):
        if self.sos is None:
            return x
        y, self.zi = sosfilt(self.sos, x, zi=self.zi)
        return y.astype(F32)


# --------------------------------------------------------------------------
# chain
# --------------------------------------------------------------------------
class VoiceChain:
    """gate -> pitch -> spectral (formant/robot/breath) -> vibrato -> ring -> drive
    -> crush -> EQ -> chorus -> echo -> reverb -> gain -> limiter"""

    def __init__(self, sr):
        self.sr = sr
        self.gate = NoiseGate(sr)
        self.pitch = PitchShifter(sr)
        self.spectral = SpectralVoice(sr)
        self.vibrato = Vibrato(sr)
        self.ring = RingMod(sr)
        self.crush = BitCrusher()
        self.eq = EQ(sr)
        self.chorus = Chorus(sr)
        self.echo = Echo(sr)
        self.reverb = Reverb(sr)
        self.limiter = Limiter()
        self.p = {}
        self.gain = 1.0

    def configure(self, p):
        self.p = dict(p)
        pitch = p.get("pitch", 0)
        self.pitch.ratio = 2 ** (pitch / 12)
        # the delay-line shifter moves formants along with pitch; correct the rest
        shifted = pitch if abs(pitch) > 0.05 else 0.0
        self.spectral.configure(p.get("formant", pitch) - shifted, p.get("robot_mix", 0),
                                p.get("robot_pitch", 120), p.get("breath", 0))
        self.eq.configure(p.get("low_cut", 0), p.get("high_cut", 20000),
                          p.get("bass", 0), p.get("treble", 0))
        self.gain = 10 ** (p.get("gain", 0) / 20)

    def set_gate(self, threshold_db):
        self.gate.threshold_db = threshold_db

    def process(self, x):
        p = self.p
        x = self.gate.process(x)
        if abs(p.get("pitch", 0)) > 0.05:
            x = self.pitch.process(x)
        if self.spectral.active:
            x = self.spectral.process(x)
        if p.get("tremble", 0) > 0.001:
            x = self.vibrato.process(x, p["tremble"])
        if p.get("ring_mix", 0) > 0.001:
            x = self.ring.process(x, p["ring_freq"], p["ring_mix"])
        drive = p.get("drive", 0)
        if drive > 0.001:
            k = 1 + 30 * drive
            x = (np.tanh(x * k) * (0.25 / math.tanh(0.25 * k))).astype(F32)
        if p.get("crush_bits", 16) < 16 or p.get("crush_down", 1) > 1:
            x = self.crush.process(x, int(p.get("crush_bits", 16)), int(p.get("crush_down", 1)))
        x = self.eq.process(x)
        if p.get("chorus_mix", 0) > 0.001:
            x = self.chorus.process(x, p["chorus_mix"], p["chorus_rate"],
                                    p["chorus_delay"], p["chorus_depth"])
        if p.get("echo_mix", 0) > 0.001:
            x = self.echo.process(x, p["echo_mix"], p["echo_time"], p["echo_fb"])
        if p.get("reverb_mix", 0) > 0.001:
            x = self.reverb.process(x, p["reverb_mix"], p["reverb_size"])
        if self.gain != 1.0:
            x = x * F32(self.gain)
        return self.limiter.process(x.astype(F32, copy=False))
