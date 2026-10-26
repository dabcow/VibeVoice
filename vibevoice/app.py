"""VibeVoice GUI (tkinter)."""
import copy
import json
import math
import os
import sys
import threading
import tkinter as tk
import webbrowser
from tkinter import messagebox, simpledialog, ttk

from . import engine as eng
from . import sharecode
from .hotkeys import MOD_ALT, MOD_CONTROL, GlobalHotkeys
from .profiles import BUILTIN, DEFAULTS, PARAMS

APP = "VibeVoice"
CFG_DIR = os.path.join(os.environ.get("APPDATA", os.path.expanduser("~")), APP)
CFG_FILE = os.path.join(CFG_DIR, "settings.json")
VBCABLE_URL = "https://vb-audio.com/Cable/"
# bundled files live next to the package, or in PyInstaller's unpack dir
ASSETS = os.path.join(getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "assets")
NONE = "(none)"
LATENCIES = {"Low (256)": 256, "Balanced (512)": 512, "Safe (1024)": 1024}
TEST_SECONDS = 5

BG, PANEL, CARD, CARD_HI = "#0f1117", "#161922", "#1f2330", "#2a2f40"
ACCENT, TEXT, MUTED, OK, BAD = "#7c5cff", "#e5e7eb", "#9ca3af", "#22c55e", "#ef4444"
FONT = "Segoe UI"


# --------------------------------------------------------------------------
# small widgets
# --------------------------------------------------------------------------
class ScrollFrame(tk.Frame):
    def __init__(self, parent, bg=PANEL, **kw):
        super().__init__(parent, bg=bg, **kw)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.bar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.bar.pack(side="right", fill="y")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(
            self._win, width=e.width))
        self.canvas.bind("<Enter>", lambda e: self.canvas.bind_all("<MouseWheel>", self._wheel))
        self.canvas.bind("<Leave>", lambda e: self.canvas.unbind_all("<MouseWheel>"))

    def _wheel(self, e):
        if self.inner.winfo_height() > self.canvas.winfo_height():
            self.canvas.yview_scroll(int(-e.delta / 120), "units")


class Toggle(tk.Label):
    """Flat pill toggle button."""

    def __init__(self, parent, text_on, text_off, command, on=False, color=ACCENT):
        super().__init__(parent, font=(FONT, 10, "bold"), padx=14, pady=7, cursor="hand2")
        self.text_on, self.text_off, self.command, self.color = text_on, text_off, command, color
        self.on = on
        self.bind("<Button-1>", lambda e: self.set(not self.on, fire=True))
        self._draw()

    def set(self, on, fire=False):
        self.on = on
        self._draw()
        if fire:
            self.command(on)

    def _draw(self):
        if self.on:
            self.configure(text=self.text_on, bg=self.color, fg="white")
        else:
            self.configure(text=self.text_off, bg=CARD, fg=MUTED)


def flat_button(parent, text, command, bg=CARD, fg=TEXT, **kw):
    b = tk.Label(parent, text=text, bg=bg, fg=fg, font=(FONT, 10), padx=12, pady=6,
                 cursor="hand2", **kw)
    b.bind("<Button-1>", lambda e: b.cget("state") != "disabled" and command())
    b.bind("<Enter>", lambda e: b.cget("state") != "disabled" and b.configure(bg=CARD_HI))
    b.bind("<Leave>", lambda e: b.configure(bg=bg))
    return b


class Meter(tk.Canvas):
    def __init__(self, parent, label):
        super().__init__(parent, width=150, height=14, bg=PANEL, highlightthickness=0)
        self.label = label
        self.level = 0.0
        self.bar = self.create_rectangle(0, 3, 0, 11, fill=OK, width=0)
        self.create_rectangle(0, 3, 150, 11, outline=CARD_HI)

    def update_peak(self, peak):
        db = 20 * math.log10(peak + 1e-9)
        target = max(0.0, min(1.0, (db + 60) / 60))
        self.level = target if target > self.level else self.level * 0.85 + target * 0.15
        color = BAD if db > -1 else ("#eab308" if db > -9 else OK)
        self.coords(self.bar, 0, 3, int(150 * self.level), 11)
        self.itemconfigure(self.bar, fill=color)


# --------------------------------------------------------------------------
# app
# --------------------------------------------------------------------------
class VibeVoiceApp:
    def __init__(self, root):
        self.root = root
        self.engine = eng.AudioEngine()
        self.cfg = self._load_cfg()
        self.custom = self.cfg.get("custom", [])
        for p in self.custom:  # voices saved before formant existed followed pitch
            p["params"].setdefault("formant", p["params"].get("pitch", 0))
        self.profiles = []
        self.current = None  # index into self.profiles
        self.work = dict(DEFAULTS)  # live params (edits on top of the profile)
        self.cards = []
        self.sliders = {}
        self._suspend_sliders = False
        self.in_devs, self.out_devs = [], []

        root.title(APP)
        try:
            root.iconbitmap(default=os.path.join(ASSETS, "icon.ico"))
        except tk.TclError:
            pass
        root.configure(bg=BG)
        root.geometry(self.cfg.get("geometry", "1180x760"))
        root.minsize(980, 640)
        self._style()
        self._build()
        self._rebuild_profiles()
        self._populate_devices(initial=True)
        self._select(self._profile_index(self.cfg.get("profile", "Clean")))
        self.engine.set_gate(self.gate_var.get())
        self._restart_engine()

        self.hotkeys = GlobalHotkeys(
            [(MOD_CONTROL | MOD_ALT, 0x31 + i, ("profile", i)) for i in range(9)]
            + [(MOD_CONTROL | MOD_ALT, 0x30, ("toggle", None)),
               (MOD_CONTROL | MOD_ALT, 0x4D, ("mute", None))])
        root.bind("<Key>", self._on_key)
        root.protocol("WM_DELETE_WINDOW", self._quit)
        self._tick()

    # ---- config ------------------------------------------------------------------
    def _load_cfg(self):
        try:
            with open(CFG_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _save_cfg(self):
        self.cfg.update({
            "custom": self.custom,
            "profile": self.profiles[self.current]["name"] if self.current is not None else "Clean",
            "hostapi": self.host_var.get(), "mic": self.mic_var.get(),
            "cable": self.cable_var.get(), "monitor": self.mon_var.get(),
            "latency": self.lat_var.get(), "gate": self.gate_var.get(),
            "monitor_vol": self.mvol_var.get(), "geometry": self.root.geometry(),
        })
        try:
            os.makedirs(CFG_DIR, exist_ok=True)
            with open(CFG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.cfg, f, indent=1)
        except OSError:
            pass

    # ---- style -------------------------------------------------------------------
    def _style(self):
        s = ttk.Style(self.root)
        s.theme_use("clam")
        s.configure(".", background=PANEL, foreground=TEXT, font=(FONT, 10))
        s.configure("TCombobox", fieldbackground=CARD, background=CARD, foreground=TEXT,
                    arrowcolor=TEXT, bordercolor=CARD_HI, lightcolor=CARD, darkcolor=CARD,
                    padding=4)
        s.map("TCombobox", fieldbackground=[("readonly", CARD)],
              foreground=[("readonly", TEXT)], selectbackground=[("readonly", CARD)],
              selectforeground=[("readonly", TEXT)])
        self.root.option_add("*TCombobox*Listbox.background", CARD)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.root.option_add("*TCombobox*Listbox.font", (FONT, 10))
        s.configure("TNotebook", background=PANEL, borderwidth=0, tabmargins=0)
        s.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(16, 7),
                    borderwidth=0, font=(FONT, 10, "bold"))
        s.map("TNotebook.Tab", background=[("selected", CARD)],
              foreground=[("selected", TEXT)])
        s.configure("Horizontal.TProgressbar", troughcolor=CARD, background=ACCENT,
                    bordercolor=CARD, lightcolor=ACCENT, darkcolor=ACCENT)
        s.configure("Vertical.TScrollbar", background=CARD, troughcolor=PANEL,
                    bordercolor=PANEL, arrowcolor=MUTED, lightcolor=CARD, darkcolor=CARD)

    def _label(self, parent, text, size=10, bold=False, fg=TEXT, bg=PANEL, **kw):
        return tk.Label(parent, text=text, bg=bg, fg=fg,
                        font=(FONT, size, "bold" if bold else "normal"), **kw)

    # ---- layout ------------------------------------------------------------------
    def _build(self):
        root = self.root
        # header
        head = tk.Frame(root, bg=PANEL, padx=16, pady=10)
        head.pack(fill="x")
        self._label(head, "VibeVoice", 16, True).pack(side="left")
        self._label(head, "  voice changer", 10, fg=MUTED).pack(side="left", pady=(6, 0))
        right = tk.Frame(head, bg=PANEL)
        right.pack(side="right")
        self.t_voice = Toggle(right, "VOICE ON", "VOICE OFF", self._on_voice_toggle, on=True)
        self.t_mon = Toggle(right, "HEARING MYSELF", "HEAR MYSELF", self._on_monitor, color="#0ea5e9")
        self.t_mute = Toggle(right, "MUTED", "MUTE", self._on_mute, color=BAD)
        for t in (self.t_mute, self.t_mon, self.t_voice):
            t.pack(side="right", padx=4)
        meters = tk.Frame(right, bg=PANEL)
        meters.pack(side="right", padx=(0, 14))
        self.m_in, self.m_out = Meter(meters, "IN"), Meter(meters, "OUT")
        for lbl, m in (("MIC", self.m_in), ("OUT", self.m_out)):
            row = tk.Frame(meters, bg=PANEL)
            row.pack(anchor="e")
            self._label(row, lbl, 8, True, fg=MUTED, width=4, anchor="e").pack(side="left")
            m.pack(side="left", padx=4, pady=1)

        # banner (shown when no virtual cable is found)
        self.banner = tk.Frame(root, bg="#3b2a0b", padx=16, pady=8)
        self._label(self.banner, "No virtual microphone driver found. Install the free "
                    "VB-Audio Virtual Cable, then restart VibeVoice — your changed voice will "
                    "appear in other apps as “CABLE Output”.", 10, fg="#fde68a",
                    bg="#3b2a0b", wraplength=820, justify="left").pack(side="left")
        flat_button(self.banner, "Get VB-CABLE (free)", lambda: webbrowser.open(VBCABLE_URL),
                    bg="#b45309", fg="white").pack(side="right")

        body = tk.Frame(root, bg=BG, padx=12, pady=12)
        body.pack(fill="both", expand=True)
        self.body = body
        body.columnconfigure(0, weight=5, uniform="c")
        body.columnconfigure(1, weight=6, uniform="c")
        body.rowconfigure(0, weight=1)

        # left: voices
        left = tk.Frame(body, bg=PANEL, padx=12, pady=12)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        top = tk.Frame(left, bg=PANEL)
        top.pack(fill="x", pady=(0, 8))
        self._label(top, "Voices", 13, True).pack(side="left")
        flat_button(top, "⬇  Import code", self._import_code, bg=CARD_HI).pack(side="right")
        self._label(top, "Ctrl+Alt+1-9 from any app", 9, fg=MUTED).pack(side="right", padx=10)
        self.voice_scroll = ScrollFrame(left)
        self.voice_scroll.pack(fill="both", expand=True)
        for c in range(3):
            self.voice_scroll.inner.columnconfigure(c, weight=1, uniform="v")

        # right: test + tabs
        rightcol = tk.Frame(body, bg=BG)
        rightcol.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self._build_test(rightcol)
        nb = ttk.Notebook(rightcol)
        nb.pack(fill="both", expand=True, pady=(12, 0))
        self.tab_custom = tk.Frame(nb, bg=CARD)
        self.tab_setup = tk.Frame(nb, bg=CARD)
        nb.add(self.tab_custom, text="Customize")
        nb.add(self.tab_setup, text="Setup")
        self._build_customize(self.tab_custom)
        self._build_setup(self.tab_setup)

        # status bar
        self.status = self._label(root, "", 9, fg=MUTED, bg=BG, anchor="w", padx=16, pady=4)
        self.status.pack(fill="x", side="bottom")

    def _build_test(self, parent):
        box = tk.Frame(parent, bg=PANEL, padx=14, pady=12)
        box.pack(fill="x")
        self._label(box, "Test your voice", 12, True).grid(row=0, column=0, sticky="w")
        self._label(box, f"Record {TEST_SECONDS}s, then play it back with any voice. "
                    "Use headphones for Hear Myself.", 9, fg=MUTED).grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(0, 8))
        self.b_rec = flat_button(box, "●  Record", self._record, bg="#7f1d1d", fg="white")
        self.b_play = flat_button(box, "▶  Play with voice", self._play_fx)
        self.b_orig = flat_button(box, "▶  Original", self._play_raw)
        self.b_stop = flat_button(box, "■  Stop", self.engine.stop_playback)
        for i, b in enumerate((self.b_rec, self.b_play, self.b_orig, self.b_stop)):
            b.grid(row=2, column=i, sticky="w", padx=(0, 8))
        self.rec_bar = ttk.Progressbar(box, maximum=1.0, length=200)
        self.rec_bar.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        box.columnconfigure(3, weight=1)

    def _build_customize(self, parent):
        top = tk.Frame(parent, bg=CARD, padx=12, pady=10)
        top.pack(fill="x")
        self.edit_title = self._label(top, "", 12, True, bg=CARD)
        self.edit_title.pack(side="left")
        for text, cmd in (("Delete", self._delete_profile), ("Reset", self._reset_profile),
                          ("Save as new", self._save_as)):
            flat_button(top, text, cmd, bg=CARD_HI).pack(side="right", padx=(6, 0))
        flat_button(top, "⬆  Share", self._share_profile, bg=ACCENT, fg="white").pack(
            side="right", padx=(6, 0))
        sf = ScrollFrame(parent, bg=CARD)
        sf.pack(fill="both", expand=True, padx=(12, 0), pady=(0, 10))
        inner = sf.inner
        inner.columnconfigure(1, weight=1)
        row, group = 0, None
        for key, label, lo, hi, res, unit, grp in PARAMS:
            if grp != group:
                group = grp
                self._label(inner, grp.upper(), 9, True, fg=ACCENT, bg=CARD).grid(
                    row=row, column=0, sticky="w", pady=(10, 2))
                row += 1
            self._label(inner, label, 10, bg=CARD, anchor="w", width=16).grid(
                row=row, column=0, sticky="w")
            var = tk.DoubleVar(value=DEFAULTS[key])
            sc = tk.Scale(inner, from_=lo, to=hi, resolution=res, orient="horizontal",
                          variable=var, showvalue=0, bg=CARD, troughcolor=BG, fg=TEXT,
                          activebackground=ACCENT, highlightthickness=0, bd=0,
                          sliderrelief="flat", sliderlength=16, width=10,
                          command=lambda v, k=key: self._on_slider(k))
            sc.grid(row=row, column=1, sticky="ew", padx=8, pady=3)
            val = self._label(inner, "", 9, fg=MUTED, bg=CARD, width=9, anchor="e")
            val.grid(row=row, column=2, sticky="e", padx=(0, 12))
            self.sliders[key] = (var, val, unit, res)
            row += 1

    def _build_setup(self, parent):
        f = tk.Frame(parent, bg=CARD, padx=16, pady=14)
        f.pack(fill="both", expand=True)
        f.columnconfigure(1, weight=1)
        self.host_var, self.mic_var = tk.StringVar(), tk.StringVar()
        self.cable_var, self.mon_var = tk.StringVar(), tk.StringVar()
        self.lat_var = tk.StringVar(value=self.cfg.get("latency", "Balanced (512)"))
        rows = [
            ("Microphone", self.mic_var, "mic"),
            ("Virtual mic output", self.cable_var, "cable"),
            ("Headphones (monitor)", self.mon_var, "mon"),
            ("Audio driver", self.host_var, "host"),
            ("Latency (buffer)", self.lat_var, "lat"),
        ]
        self.combos = {}
        for r, (label, var, key) in enumerate(rows):
            self._label(f, label, 10, bg=CARD).grid(row=r, column=0, sticky="w", pady=5)
            cb = ttk.Combobox(f, textvariable=var, state="readonly")
            cb.grid(row=r, column=1, sticky="ew", pady=5, padx=(12, 0))
            cb.bind("<<ComboboxSelected>>", lambda e, k=key: self._on_device_change(k))
            self.combos[key] = cb
        self.combos["host"]["values"] = eng.host_apis()
        self.combos["lat"]["values"] = list(LATENCIES)

        r = len(rows)
        self.gate_var = tk.DoubleVar(value=self.cfg.get("gate", -50))
        self.mvol_var = tk.DoubleVar(value=self.cfg.get("monitor_vol", 0.8))
        for label, var, lo, hi, res, cmd in (
                ("Noise gate (dB)", self.gate_var, -80, -20, 1, self._on_gate),
                ("Monitor volume", self.mvol_var, 0, 1.5, 0.05, self._on_mvol)):
            self._label(f, label, 10, bg=CARD).grid(row=r, column=0, sticky="w", pady=5)
            tk.Scale(f, from_=lo, to=hi, resolution=res, orient="horizontal", variable=var,
                     bg=CARD, fg=MUTED, troughcolor=BG, activebackground=ACCENT,
                     highlightthickness=0, bd=0, sliderrelief="flat", sliderlength=16,
                     width=10, font=(FONT, 8), command=lambda v, c=cmd: c()).grid(
                row=r, column=1, sticky="ew", padx=(12, 0))
            r += 1
        self.engine.monitor_vol = self.mvol_var.get()

        help_text = (
            "How to use your new voice in other apps\n"
            "1. Install VB-Audio Virtual Cable (free) once and reboot if asked.\n"
            "2. Set 'Virtual mic output' above to “CABLE Input”.\n"
            "3. In Discord / Zoom / OBS / games pick “CABLE Output” as the microphone.\n"
            "Noise gate: raise it if background noise leaks through; -80 turns it off.\n"
            "Hotkeys (work everywhere): Ctrl+Alt+1-9 voices, Ctrl+Alt+0 voice on/off, "
            "Ctrl+Alt+M mute.")
        self._label(f, help_text, 9, fg=MUTED, bg=CARD, justify="left", anchor="w",
                    wraplength=520).grid(row=r, column=0, columnspan=2, sticky="w", pady=(14, 6))
        flat_button(f, "Open VB-CABLE download page", lambda: webbrowser.open(VBCABLE_URL),
                    bg=CARD_HI).grid(row=r + 1, column=0, sticky="w")

    # ---- profiles ---------------------------------------------------------------
    def _rebuild_profiles(self):
        sel_name = self.profiles[self.current]["name"] if self.current is not None else None
        self.profiles = BUILTIN + self.custom
        for c in self.cards:
            c.destroy()
        self.cards = []
        inner = self.voice_scroll.inner
        for i, p in enumerate(self.profiles):
            card = tk.Frame(inner, bg=CARD, cursor="hand2", highlightthickness=2,
                            highlightbackground=CARD, highlightcolor=CARD)
            card.grid(row=i // 3, column=i % 3, sticky="nsew", padx=4, pady=4)
            tk.Frame(card, bg=p["color"], width=6).pack(side="left", fill="y")
            txt = tk.Frame(card, bg=CARD, padx=10, pady=10)
            txt.pack(side="left", fill="both", expand=True)
            name = self._label(txt, p["name"], 10, True, bg=CARD, anchor="w")
            name.pack(fill="x")
            hint = ("Ctrl+Alt+%d" % (i + 1)) if i < 9 else ("custom" if not p.get("builtin") else "")
            sub = self._label(txt, hint, 8, fg=MUTED, bg=CARD, anchor="w")
            sub.pack(fill="x")
            for w in (card, txt, name, sub):
                w.bind("<Button-1>", lambda e, k=i: self._select(k))
            card._parts = (txt, name, sub)
            self.cards.append(card)
        if sel_name is not None:
            self.current = None
            self._select(self._profile_index(sel_name))

    def _profile_index(self, name):
        return next((i for i, p in enumerate(self.profiles) if p["name"] == name), 0)

    def _select(self, i):
        if not 0 <= i < len(self.profiles):
            return
        if self.current is not None and self.current < len(self.cards):
            self._paint_card(self.current, False)
        self.current = i
        self._paint_card(i, True)
        self.work = dict(DEFAULTS)
        self.work.update(self.profiles[i]["params"])
        self._load_sliders()
        self._update_title()
        self.engine.set_params(self.work)

    def _paint_card(self, i, selected):
        card = self.cards[i]
        bg = CARD_HI if selected else CARD
        card.configure(bg=bg, highlightbackground=ACCENT if selected else CARD)
        for w in card._parts:
            w.configure(bg=bg)

    def _update_title(self):
        p = self.profiles[self.current]
        modified = any(abs(self.work[k] - p["params"].get(k, DEFAULTS[k])) > 1e-9 for k in DEFAULTS)
        self.edit_title.configure(text=p["name"] + ("  • modified" if modified else ""))

    def _load_sliders(self):
        self._suspend_sliders = True
        for key, (var, val, unit, res) in self.sliders.items():
            var.set(self.work[key])
            val.configure(text=self._fmt(self.work[key], unit, res, key))
        self._suspend_sliders = False

    @staticmethod
    def _fmt(v, unit, res, key=""):
        if key == "target_f0" and v <= 0:
            return "off"
        s = f"{v:+.1f}" if unit in ("st", "dB") else (f"{v:.0f}" if res >= 1 else f"{v:.2f}")
        return f"{s} {unit}".strip()

    def _on_slider(self, key):
        if self._suspend_sliders:
            return
        var, val, unit, res = self.sliders[key]
        self.work[key] = round(var.get(), 4)
        val.configure(text=self._fmt(self.work[key], unit, res, key))
        self.engine.set_params(self.work)
        p = self.profiles[self.current]
        if not p.get("builtin"):  # custom profiles save edits automatically
            p["params"] = dict(self.work)
        self._update_title()

    def _save_as(self):
        base = self.profiles[self.current]["name"]
        name = simpledialog.askstring(APP, "Name for the new voice:", parent=self.root,
                                      initialvalue=f"My {base}")
        if not name:
            return
        name = name.strip()[:28]
        if any(p["name"].lower() == name.lower() for p in self.profiles):
            messagebox.showerror(APP, f"A voice named “{name}” already exists.")
            return
        src = self.profiles[self.current]
        self.custom.append({"name": name, "color": src["color"], "params": dict(self.work),
                            "builtin": False, "base": src.get("base", src["name"])})
        self._rebuild_profiles()
        self._select(self._profile_index(name))
        self._save_cfg()

    def _reset_profile(self):
        p = self.profiles[self.current]
        if not p.get("builtin"):
            builtin = next((b for b in BUILTIN if b["name"] == p.get("base")), None)
            if builtin is None or not messagebox.askyesno(
                    APP, f"Reset “{p['name']}” to the “{builtin['name']}” preset?"):
                return
            p["params"] = copy.deepcopy(builtin["params"])
        self._select(self.current)

    # ---- share codes ---------------------------------------------------------------
    def _code_dialog(self, title, message, code=None):
        """Dark modal with a code field. Returns the entered text on OK (import mode)."""
        dlg = tk.Toplevel(self.root, bg=PANEL, padx=18, pady=16)
        dlg.title(title)
        dlg.transient(self.root)
        dlg.resizable(False, False)
        self._label(dlg, title, 12, True).pack(anchor="w")
        self._label(dlg, message, 9, fg=MUTED, justify="left", wraplength=460).pack(
            anchor="w", pady=(2, 10))
        var = tk.StringVar(value=code or "")
        entry = tk.Entry(dlg, textvariable=var, width=58, font=("Consolas", 11), bg=CARD,
                         fg=TEXT, insertbackground=TEXT, relief="flat",
                         highlightthickness=1, highlightbackground=CARD_HI,
                         highlightcolor=ACCENT)
        entry.pack(fill="x", ipady=6)
        status = self._label(dlg, "", 9, fg=OK)
        status.pack(anchor="w", pady=(6, 0))
        row = tk.Frame(dlg, bg=PANEL)
        row.pack(fill="x", pady=(10, 0))
        result = {}

        def copy():
            self.root.clipboard_clear()
            self.root.clipboard_append(var.get())
            status.configure(text="✓ Copied to clipboard", fg=OK)

        def ok():
            result["code"] = var.get()
            dlg.destroy()

        if code is not None:  # share mode
            entry.configure(state="readonly", readonlybackground=CARD)
            flat_button(row, "Close", dlg.destroy, bg=CARD_HI).pack(side="right")
            flat_button(row, "Copy code", copy, bg=ACCENT, fg="white").pack(side="right", padx=6)
            copy()
        else:
            flat_button(row, "Cancel", dlg.destroy, bg=CARD_HI).pack(side="right")
            flat_button(row, "Add voice", ok, bg=ACCENT, fg="white").pack(side="right", padx=6)
            entry.bind("<Return>", lambda e: ok())
            try:  # pre-fill when the clipboard already holds a code
                clip = self.root.clipboard_get().strip()
                if clip.upper().startswith(sharecode.PREFIX.upper()):
                    var.set(clip)
            except tk.TclError:
                pass
        entry.bind("<Escape>", lambda e: dlg.destroy())
        entry.focus_set()
        entry.select_range(0, "end")
        dlg.geometry(f"+{self.root.winfo_rootx() + 120}+{self.root.winfo_rooty() + 120}")
        try:  # modal; grabbing needs the window to be on screen first
            dlg.wait_visibility()
            dlg.grab_set()
        except tk.TclError:
            pass
        self.root.wait_window(dlg)
        return result.get("code")

    def _share_profile(self):
        p = self.profiles[self.current]
        code = sharecode.encode(p["name"], p["color"], self.work)
        self._code_dialog(f"Share “{p['name']}”",
                          "Send this code to a friend. They can click Import code in VibeVoice "
                          "and paste it to get this exact voice, including any unsaved slider "
                          "changes.", code)

    def _import_code(self):
        text = self._code_dialog("Import a voice",
                                 "Paste a voice code someone shared with you (it starts "
                                 "with VV1-).")
        if not text:
            return
        try:
            shared = sharecode.decode(text)
        except ValueError as ex:
            messagebox.showerror(APP, str(ex))
            return
        name, n = shared["name"], 2
        while any(p["name"].lower() == name.lower() for p in self.profiles):
            name = f"{shared['name']} ({n})"
            n += 1
        self.custom.append({"name": name, "color": shared["color"],
                            "params": shared["params"], "builtin": False})
        self._rebuild_profiles()
        self._select(self._profile_index(name))
        self._save_cfg()

    def _delete_profile(self):
        p = self.profiles[self.current]
        if p.get("builtin"):
            messagebox.showinfo(APP, "Built-in voices can't be deleted.")
            return
        if messagebox.askyesno(APP, f"Delete “{p['name']}”?"):
            self.custom.remove(p)
            self.current = None
            self._rebuild_profiles()
            self._select(0)
            self._save_cfg()

    # ---- devices -------------------------------------------------------------------
    def _populate_devices(self, initial=False):
        apis = eng.host_apis()
        host = self.cfg.get("hostapi") if initial else self.host_var.get()
        if host not in apis:
            host = "Windows WASAPI" if "Windows WASAPI" in apis else apis[0]
        self.host_var.set(host)
        self.engine.hostapi = host
        self.in_devs, self.out_devs = eng.list_devices(host)
        in_names = [n for _, n in self.in_devs]
        out_names = [n for _, n in self.out_devs]
        self.combos["mic"]["values"] = in_names
        self.combos["cable"]["values"] = [NONE] + out_names
        self.combos["mon"]["values"] = [NONE] + out_names
        d_in, d_out = eng.default_devices(host)
        name_of = {i: n for i, n in self.in_devs + self.out_devs}

        def pick(saved, names, fallback):
            if initial and saved in names:
                return saved
            return fallback

        mic_default = name_of.get(d_in)
        if not mic_default or eng.is_virtual(mic_default):
            mic_default = next((n for n in in_names if not eng.is_virtual(n)),
                               in_names[0] if in_names else "")
        self.mic_var.set(pick(self.cfg.get("mic"), in_names, mic_default))

        cable_default = next((n for n in out_names if "cable input" in n.lower()),
                             next((n for n in out_names if eng.is_virtual(n)), NONE))
        self.cable_var.set(pick(self.cfg.get("cable"), [NONE] + out_names, cable_default))

        mon_default = name_of.get(d_out)
        if not mon_default or eng.is_virtual(mon_default):
            mon_default = next((n for n in out_names if not eng.is_virtual(n)), NONE)
        self.mon_var.set(pick(self.cfg.get("monitor"), [NONE] + out_names, mon_default))

        has_virtual = any(eng.is_virtual(n) for n in out_names)
        if has_virtual:
            self.banner.pack_forget()
        else:
            self.banner.pack(fill="x", before=self.body)

    def _dev_index(self, name, devs):
        return next((i for i, n in devs if n == name), None)

    def _on_device_change(self, key):
        if key == "host":
            self._populate_devices()
        if self.cable_var.get() != NONE and self.cable_var.get() == self.mon_var.get():
            self.mon_var.set(NONE)
        self._restart_engine()
        self._save_cfg()

    def _restart_engine(self):
        e = self.engine
        e.blocksize = LATENCIES.get(self.lat_var.get(), 512)
        mic = self._dev_index(self.mic_var.get(), self.in_devs)
        cable = self._dev_index(self.cable_var.get(), self.out_devs)
        mon = self._dev_index(self.mon_var.get(), self.out_devs)
        errors = e.start(mic, cable, mon)
        e.set_params(self.work)
        if self.t_mon.on:
            self._on_monitor(True)
        if errors:
            messagebox.showwarning(APP, "Some audio devices could not be opened:\n\n"
                                   + "\n".join(errors)
                                   + "\n\nTry another device, the MME driver, or a safer latency.")

    # ---- controls ----------------------------------------------------------------------
    def _on_voice_toggle(self, on):
        self.engine.enabled = on

    def _on_mute(self, on):
        self.engine.muted = on

    def _on_monitor(self, on):
        try:
            self.engine.set_monitor(on)
            if on and self.engine.mon_stream is None:
                raise RuntimeError("Choose a headphones device in Setup first.")
        except Exception as ex:  # noqa: BLE001
            self.t_mon.set(False)
            messagebox.showwarning(APP, f"Can't start monitoring:\n{ex}")

    def _on_gate(self):
        self.engine.set_gate(self.gate_var.get())

    def _on_mvol(self):
        self.engine.monitor_vol = self.mvol_var.get()

    def _record(self):
        if not self.engine.running:
            messagebox.showwarning(APP, "The microphone isn't running. Check Setup.")
            return
        self.engine.stop_playback()
        self.engine.start_recording(TEST_SECONDS)

    def _play(self, processed):
        raw = self.engine.recording
        if raw is None:
            messagebox.showinfo(APP, "Record a test clip first.")
            return
        params = dict(self.work)

        def work():
            try:
                audio = self.engine.render(raw, params) if processed else raw
                self.engine.play(audio)
            except Exception as ex:  # noqa: BLE001
                self.root.after(0, lambda: messagebox.showwarning(APP, str(ex)))
        threading.Thread(target=work, daemon=True).start()

    def _play_fx(self):
        self._play(True)

    def _play_raw(self):
        self._play(False)

    def _on_key(self, e):
        if isinstance(e.widget, (tk.Entry, ttk.Combobox)):
            return
        if e.char and e.char in "123456789":
            self._select(int(e.char) - 1)

    def _handle_hotkey(self, tag, arg):
        if tag == "profile":
            self._select(arg)
        elif tag == "toggle":
            self.t_voice.set(not self.t_voice.on, fire=True)
        elif tag == "mute":
            self.t_mute.set(not self.t_mute.on, fire=True)

    # ---- periodic UI update ------------------------------------------------------------------
    def _tick(self):
        e = self.engine
        a, b = e.take_peaks()
        self.m_in.update_peak(a)
        self.m_out.update_peak(b)
        prog = e.rec_progress
        if prog is not None:
            self.rec_bar["value"] = prog
            self.b_rec.configure(text=f"●  Recording… {TEST_SECONDS * (1 - prog):.0f}s")
        elif self.b_rec.cget("text") != "●  Record":
            self.rec_bar["value"] = 1.0 if e.recording is not None else 0
            self.b_rec.configure(text="●  Record")
        while not self.hotkeys.events.empty():
            self._handle_hotkey(*self.hotkeys.events.get_nowait())
        if e.running:
            lat = (e.in_stream.latency or 0) + 2 * e.blocksize / eng.SR
            if e.cable_stream is not None:
                lat += e.cable_stream.latency or 0
            sv = e.chain.spectral
            if sv.active:
                lat += sv.N / eng.SR
            out = self.cable_var.get() if e.cable_stream is not None else "no virtual mic"
            heard = ""
            if self.work.get("target_f0", 0) > 0:
                heard = (f"   your pitch ~{sv.user_f0:.0f} Hz" if sv.user_f0
                         else "   learning your pitch… keep talking")
            self.status.configure(
                text=f"● Running   DSP load {e.load * 100:4.1f}%   "
                     f"latency ~{lat * 1000:.0f} ms   output → {out}" + heard
                     + (f"   glitches {e.xruns}" if e.xruns else ""), fg=OK)
        else:
            self.status.configure(text="● Stopped — check the microphone in Setup", fg=BAD)
        self.root.after(50, self._tick)

    def _quit(self):
        self._save_cfg()
        self.hotkeys.stop()
        self.engine.stop_playback()
        self.engine.stop()
        self.root.destroy()


def main():
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # crisp text on high-DPI screens
        # own taskbar identity, so Windows shows our icon instead of Python's
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("VibeVoice.App")
    except (AttributeError, OSError):
        pass
    root = tk.Tk()
    VibeVoiceApp(root)
    root.mainloop()
