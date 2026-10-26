"""Real-time voice effects.

Every effect works on mono float32 blocks and keeps its own state, so the
chain can be fed arbitrary block sizes. All per-sample work is vectorised
with numpy; there are no Python-level per-sample loops.
"""
import math

import numpy as np

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
    The damping lowpass is a 2-tap FIR, so it stays vectorised.
    """

    def __init__(self, max_delay):
        self.ring = _Ring(max_delay + 1)
        self.last = 0.0

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
            prev = np.concatenate(([self.last], d[:-1]))
            self.last = float(d[-1])
            self.ring.write(x + fb * ((1.0 - damp) * d + damp * prev))
        else:
            self.ring.write(x + fb * d)
        return d

    def clear(self):
        self.ring.clear()
        self.last = 0.0


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
    """STFT stage: pitch + formant shifting, monotone robot and breathiness.

    Pitch uses a phase-locked phase vocoder (Laroche & Dolson): spectral peaks
    are moved to their new frequency with coherent phase, after the voice is
    flattened by its cepstral envelope. The envelope is then re-applied at the
    formant position asked for, so pitch and timbre move independently.
    With `target_f0` set, a pitch tracker learns the speaker's average pitch and
    shifts that average to the target, keeping natural intonation.
    Adds one frame (N samples, ~43 ms) of latency, only while active.
    """

    N, HOP, LIFTER = 2048, 256, 180
    DEFAULT_F0 = 120.0

    def __init__(self, sr):
        self.sr = sr
        n, hop = self.N, self.HOP
        nb = n // 2 + 1
        self.win = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n))
        self.ola_scale = hop / (n * 0.5)  # periodic Hann (win^2) sums to n / (2 * hop)
        self.frame = np.zeros(n)
        self.ola = np.zeros(n)
        self.inq = np.zeros(0, F32)
        self.outq = np.zeros(hop, F32)  # one hop of priming keeps output whole
        lift = np.zeros(n)
        lift[0] = 1.0
        lift[1:self.LIFTER] = 2.0  # fold negative quefrencies onto positive
        self.lifter = lift
        self.bins = np.arange(nb, dtype=np.float64)
        self.ibins = np.arange(nb)
        self.omega = 2 * np.pi * self.bins / n
        self.prev_ph = np.zeros(nb)
        self.acc = np.zeros(nb)
        self.rng = np.random.default_rng()
        # pitch tracker (autocorrelation of the analysis frame)
        self.wac = np.fft.irfft(np.abs(np.fft.rfft(self.win)) ** 2, n)
        self.lag_lo, self.lag_hi = sr // 450, sr // 65
        self.avg_lf0 = math.log2(self.DEFAULT_F0)
        self.f0_count = 0
        # robot carrier
        self.car_frame = np.zeros(n)
        self.car_phase = 0.0
        self.car_f0 = None
        self.table = None
        # params
        self.pv = False
        self.pitch_st = self.formant_st = 0.0
        self.target_f0 = 0.0
        self.robot = 0.0
        self.robot_f0 = 120.0
        self.breath = 0.0

    def configure(self, pv, pitch_st, formant_st, target_f0, robot, robot_f0, breath):
        self.pv = pv
        self.pitch_st, self.formant_st, self.target_f0 = pitch_st, formant_st, target_f0
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
        return self.pv or self.robot > 0.001 or self.breath > 0.001

    @property
    def user_f0(self):
        """Learned average speaking pitch in Hz (None until enough voiced audio)."""
        return 2 ** self.avg_lf0 if self.f0_count > 20 else None

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

    def _track_pitch(self, power):
        ac = np.fft.irfft(power, self.N)
        if ac[0] < 1e-6 * self.wac[0]:  # too quiet to judge
            return
        r = (ac[:self.lag_hi] / self.wac[:self.lag_hi]) * (self.wac[0] / ac[0])
        seg = r[self.lag_lo:self.lag_hi]
        top = float(seg.max())
        if top < 0.6:  # unvoiced
            return
        i = int(np.flatnonzero(seg >= 0.85 * top)[0])  # shortest strong lag: no octave-down errors
        while i + 1 < len(seg) and seg[i + 1] > seg[i]:
            i += 1
        lf0 = math.log2(self.sr / (self.lag_lo + i))
        if self.f0_count > 20 and abs(lf0 - self.avg_lf0) > 0.9:
            return  # outlier (likely an octave error)
        self.f0_count += 1
        alpha = max(1.0 / self.f0_count, self.HOP / (self.sr * 3.0))
        self.avg_lf0 += (lf0 - self.avg_lf0) * alpha

    def _shift_amounts(self):
        """(pitch ratio, formant ratio) for this hop."""
        if self.target_f0 > 0:
            auto = 12 * (math.log2(self.target_f0) - self.avg_lf0)
            nominal = 12 * math.log2(self.target_f0 / self.DEFAULT_F0)
            # a speaker already near the target needs less timbre change
            scale = min(1.0, max(0.0, auto / nominal)) if abs(nominal) > 0.5 else 1.0
            st = min(19.0, max(-12.0, auto + self.pitch_st))
            return 2 ** (st / 12), 2 ** (self.formant_st * scale / 12)
        return 2 ** (self.pitch_st / 12), 2 ** (self.formant_st / 12)

    def _hop(self, chunk):
        n, hop = self.N, self.HOP
        self.frame[:-hop] = self.frame[hop:]
        self.frame[-hop:] = chunk
        X = np.fft.rfft(self.frame * self.win)
        mag = np.abs(X) + 1e-9
        env = np.fft.rfft(np.fft.irfft(np.log(mag), n) * self.lifter).real
        env_t, S = env, X
        if self.pv:
            ph = np.angle(X)
            dphi = ph - self.prev_ph - self.omega * hop
            self.prev_ph = ph
            inst = self.omega + (np.mod(dphi + np.pi, 2 * np.pi) - np.pi) / hop
            if self.target_f0 > 0:
                self._track_pitch(mag * mag)
            beta, alpha = self._shift_amounts()
            if abs(alpha - 1.0) > 1e-3:
                env_t = np.interp(self.bins / alpha, self.bins, env)
            if abs(beta - 1.0) > 1e-3:
                self.acc = np.mod(self.acc + hop * beta * inst, 2 * np.pi)
                S = self._transpose(mag, ph, env, env_t, beta)
            else:
                self.acc = ph
                if env_t is not env:
                    S = X * np.exp(np.clip(env_t - env, -4.0, 4.0))
        energy = float(np.sum(np.abs(S) ** 2))
        if self.robot > 0.001 or self.breath > 0.001:
            shape = np.exp(env_t - env_t.max())
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

    def _transpose(self, mag, ph, env, env_t, beta):
        """Phase-locked bin shifting of the flattened (envelope-free) spectrum."""
        nb = len(mag)
        m = mag[1:-1]
        pk = np.flatnonzero((m > mag[:-2]) & (m >= mag[2:]) & (m > mag.max() * 1e-3)) + 1
        if len(pk) == 0:
            return np.zeros(nb, complex)
        region = np.searchsorted((pk[:-1] + pk[1:]) // 2, self.ibins, side="right")
        shift = np.rint(pk * beta).astype(np.int64) - pk
        theta = self.acc[pk] - ph[pk]
        dst = self.ibins + shift[region]
        ok = (dst >= 0) & (dst < nb)
        vals = (mag / np.exp(env)) * np.exp(1j * (ph + theta[region]))
        S = np.zeros(nb, complex)
        S[dst[ok]] = vals[ok]
        return S * np.exp(env_t)


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
# EQ (RBJ biquad magnitudes -> one minimum-phase FIR, FFT convolution)
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
    """The combined response of all bands, realised as one minimum-phase FIR.

    Minimum phase keeps the delay near zero like the IIR filters it replaces,
    and a single FFT convolution per block is cheaper than several biquads.
    """

    TAPS, DESIGN = 2048, 16384

    def __init__(self, sr):
        self.sr = sr
        self.h = None
        self.tail = np.zeros(self.TAPS - 1)
        self._nfft = 0
        self._hf = None

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
            self.h = None
            return
        nd = self.DESIGN
        z1 = np.exp(-1j * np.linspace(0, np.pi, nd // 2 + 1))
        z2 = z1 * z1
        mag = np.ones(len(z1))
        for b0, b1, b2, _, a1, a2 in secs:
            mag *= np.abs((b0 + b1 * z1 + b2 * z2) / (1 + a1 * z1 + a2 * z2))
        # minimum phase from magnitude (homomorphic / folded cepstrum)
        c = np.fft.irfft(np.log(np.maximum(mag, 1e-5)), nd)
        c[1:nd // 2] *= 2
        c[nd // 2 + 1:] = 0
        h = np.fft.irfft(np.exp(np.fft.rfft(c)), nd)[:self.TAPS]
        fade = 256
        h[-fade:] *= 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, fade))
        self.h = h
        self._nfft = 0

    def process(self, x):
        if self.h is None:
            return x
        n, L = len(x), self.TAPS
        nfft = 1 << (n + L - 2).bit_length()
        if nfft != self._nfft:
            self._hf = np.fft.rfft(self.h, nfft)
            self._nfft = nfft
        y = np.fft.irfft(np.fft.rfft(x, nfft) * self._hf, nfft)[:n + L - 1]
        y[:L - 1] += self.tail
        self.tail = y[n:].copy()
        return y[:n].astype(F32)


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
        self.delay_pitch = False

    def configure(self, p):
        self.p = dict(p)
        pitch = p.get("pitch", 0)
        formant = p.get("formant", pitch)
        target = p.get("target_f0", 0)
        # cheap delay-line shifter when formants simply follow pitch; otherwise
        # the phase vocoder moves pitch and formants independently
        pv = target > 0 or abs(formant - pitch) > 0.05
        self.pitch.ratio = 2 ** (pitch / 12)
        self.delay_pitch = not pv and abs(pitch) > 0.05
        self.spectral.configure(pv, pitch, formant, target, p.get("robot_mix", 0),
                                p.get("robot_pitch", 120), p.get("breath", 0))
        self.eq.configure(p.get("low_cut", 0), p.get("high_cut", 20000),
                          p.get("bass", 0), p.get("treble", 0))
        self.gain = 10 ** (p.get("gain", 0) / 20)

    def set_gate(self, threshold_db):
        self.gate.threshold_db = threshold_db

    def process(self, x):
        p = self.p
        x = self.gate.process(x)
        if self.delay_pitch:
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
