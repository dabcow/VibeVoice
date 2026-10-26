"""Audio I/O: mic -> VoiceChain -> virtual cable (+ optional monitor)."""
import threading
import time

import numpy as np
import sounddevice as sd

from .dsp import F32, VoiceChain

SR = 48000
VIRTUAL_HINTS = ("cable input", "voicemeeter input", "voicemeeter aux input",
                 "vb-audio", "virtual")


class AudioFifo:
    """Thread-safe mono FIFO between two device clocks.

    Keeps a small cushion (`prefill`) to absorb jitter and drops the oldest
    audio when it builds up, so latency stays bounded despite clock drift.
    """

    def __init__(self, prefill, size=SR):
        self.buf = np.zeros(size, F32)
        self.size = size
        self.r = self.w = self.fill = 0
        self.prefill = prefill
        self.priming = True
        self.lock = threading.Lock()

    def write(self, x):
        n = len(x)
        with self.lock:
            end = self.w + n
            if end <= self.size:
                self.buf[self.w:end] = x
            else:
                k = self.size - self.w
                self.buf[self.w:] = x[:k]
                self.buf[: n - k] = x[k:]
            self.w = end % self.size
            self.fill = min(self.fill + n, self.size)
            limit = self.prefill * 4 + n
            if self.fill > limit:  # drift: drop oldest, keep cushion
                drop = self.fill - (self.prefill + n)
                self.r = (self.r + drop) % self.size
                self.fill -= drop
            elif self.fill == self.size:
                self.r = self.w

    def read(self, n):
        out = np.zeros(n, F32)
        with self.lock:
            if self.priming:
                if self.fill < n + self.prefill:
                    return out
                self.priming = False
            m = min(n, self.fill)
            end = self.r + m
            if end <= self.size:
                out[:m] = self.buf[self.r:end]
            else:
                k = self.size - self.r
                out[:k] = self.buf[self.r:]
                out[k:m] = self.buf[: m - k]
            self.r = end % self.size
            self.fill -= m
            if m < n:
                self.priming = True
        return out


def host_apis():
    return [h["name"] for h in sd.query_hostapis()]


def list_devices(hostapi_name):
    """Return (inputs, outputs) as lists of (index, name) for one host API."""
    apis = sd.query_hostapis()
    idx = next((i for i, h in enumerate(apis) if h["name"] == hostapi_name), None)
    ins, outs = [], []
    if idx is None:
        return ins, outs
    for d in sd.query_devices():
        if d["hostapi"] != idx:
            continue
        if d["max_input_channels"] > 0:
            ins.append((d["index"], d["name"]))
        if d["max_output_channels"] > 0:
            outs.append((d["index"], d["name"]))
    return ins, outs


def default_devices(hostapi_name):
    apis = sd.query_hostapis()
    h = next((h for h in apis if h["name"] == hostapi_name), None)
    if h is None:
        return None, None
    return h.get("default_input_device"), h.get("default_output_device")


def is_virtual(name):
    n = name.lower()
    return any(h in n for h in VIRTUAL_HINTS)


class AudioEngine:
    def __init__(self):
        self.chain = VoiceChain(SR)
        self.params = {}
        self.gate_db = -50.0
        self._pending = None
        self.enabled = True
        self.muted = False
        self.monitor_vol = 0.8
        self.hostapi = "Windows WASAPI"
        self.blocksize = 512

        self.in_stream = self.cable_stream = self.mon_stream = None
        self.mon_device = None
        self.cable_fifo = self.mon_fifo = None
        self._in_ch = 1

        self.in_peak = 0.0
        self.out_peak = 0.0
        self.load = 0.0
        self.xruns = 0

        self._rec = None
        self._rec_target = self._rec_n = 0
        self.recording = None

    # ---- parameters (called from GUI thread) ---------------------------------
    def set_params(self, params):
        self.params = dict(params)
        self._pending = (self.params, self.gate_db)

    def set_gate(self, db):
        self.gate_db = db
        self._pending = (self.params, db)

    # ---- streams -------------------------------------------------------------
    def _extra(self):
        if "WASAPI" in self.hostapi:
            return sd.WasapiSettings(auto_convert=True)
        return None

    def _channels(self, dev, kind):
        info = sd.query_devices(dev)
        return max(1, min(2, info[f"max_{kind}_channels"]))

    def start(self, mic, cable, monitor):
        self.stop()
        errors = []
        self.cable_fifo = AudioFifo(self.blocksize)
        self.mon_fifo = AudioFifo(self.blocksize)
        self.mon_device = monitor
        if mic is None:
            return ["No microphone selected"]
        try:
            self._in_ch = self._channels(mic, "input")
            self.in_stream = sd.InputStream(
                device=mic, samplerate=SR, blocksize=self.blocksize,
                channels=self._in_ch, dtype="float32", latency="low",
                callback=self._in_cb, extra_settings=self._extra())
            self.in_stream.start()
        except Exception as e:  # noqa: BLE001 - surface any PortAudio error
            self.in_stream = None
            errors.append(f"Microphone: {e}")
        if cable is not None:
            try:
                self.cable_stream = self._open_out(cable, self._cable_cb)
            except Exception as e:  # noqa: BLE001
                self.cable_stream = None
                errors.append(f"Virtual mic output: {e}")
        return errors

    def _open_out(self, dev, cb):
        s = sd.OutputStream(
            device=dev, samplerate=SR, blocksize=self.blocksize,
            channels=self._channels(dev, "output"), dtype="float32",
            latency="low", callback=cb, extra_settings=self._extra())
        s.start()
        return s

    def set_monitor(self, on):
        if on and self.mon_stream is None and self.mon_device is not None:
            self.mon_fifo = AudioFifo(self.blocksize)
            self.mon_stream = self._open_out(self.mon_device, self._mon_cb)
        elif not on and self.mon_stream is not None:
            s, self.mon_stream = self.mon_stream, None
            s.stop()
            s.close()

    def stop(self):
        for name in ("in_stream", "cable_stream", "mon_stream"):
            s = getattr(self, name)
            setattr(self, name, None)
            if s is not None:
                try:
                    s.stop()
                    s.close()
                except Exception:  # noqa: BLE001
                    pass

    @property
    def running(self):
        return self.in_stream is not None

    # ---- callbacks (audio threads) ---------------------------------------------
    def _in_cb(self, indata, frames, _t, status):
        t0 = time.perf_counter()
        if status:
            self.xruns += 1
        x = indata[:, 0].copy() if self._in_ch == 1 else indata.mean(axis=1, dtype=F32)
        pending = self._pending
        if pending is not None:
            self._pending = None
            self.chain.configure(pending[0])
            self.chain.set_gate(pending[1])
        if self._rec is not None:
            self._rec.append(x.copy())
            self._rec_n += frames
            if self._rec_n >= self._rec_target:
                self.recording = np.concatenate(self._rec)[: self._rec_target]
                self._rec = None
        self.in_peak = max(self.in_peak, float(np.max(np.abs(x))))
        if self.muted:
            y = np.zeros(frames, F32)
        elif self.enabled:
            y = self.chain.process(x)
        else:
            y = x
        self.out_peak = max(self.out_peak, float(np.max(np.abs(y))))
        if self.cable_stream is not None:
            self.cable_fifo.write(y)
        if self.mon_stream is not None:
            self.mon_fifo.write(y * F32(self.monitor_vol))
        used = (time.perf_counter() - t0) / (frames / SR)
        self.load += (used - self.load) * 0.05

    def _cable_cb(self, outdata, frames, _t, _status):
        outdata[:] = self.cable_fifo.read(frames)[:, None]

    def _mon_cb(self, outdata, frames, _t, _status):
        outdata[:] = self.mon_fifo.read(frames)[:, None]

    def take_peaks(self):
        a, b = self.in_peak, self.out_peak
        self.in_peak = self.out_peak = 0.0
        return a, b

    # ---- voice test ----------------------------------------------------------------
    def start_recording(self, seconds):
        self.recording = None
        self._rec_target = int(seconds * SR)
        self._rec_n = 0
        self._rec = []

    @property
    def rec_progress(self):
        if self._rec is None:
            return None
        return min(1.0, self._rec_n / max(1, self._rec_target))

    def render(self, raw, params):
        """Process a recorded clip offline with a fresh chain."""
        c = VoiceChain(SR)
        live = self.chain.spectral
        if live.user_f0:  # reuse the pitch already learned from the live mic
            c.spectral.avg_lf0, c.spectral.f0_count = live.avg_lf0, live.f0_count
        c.configure(params)
        c.set_gate(self.gate_db)
        out = np.empty_like(raw)
        for i in range(0, len(raw), self.blocksize):
            out[i:i + self.blocksize] = c.process(raw[i:i + self.blocksize])
        return out

    def play(self, audio):
        if self.mon_device is None:
            raise RuntimeError("Choose a headphones/speakers device first")
        sd.stop()
        a = audio * F32(self.monitor_vol)
        if self._channels(self.mon_device, "output") == 2:
            a = np.column_stack((a, a))
        sd.play(a, SR, device=self.mon_device,
                extra_settings=self._extra())

    def stop_playback(self):
        sd.stop()
