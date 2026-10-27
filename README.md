# VibeVoice

A free, lightweight real-time voice changer for Windows. Pick a voice and talk:
your changed voice goes to a virtual microphone that Discord, Zoom, OBS and
games can use.

## Quick start

1. Install **VB-Audio Virtual Cable** (free) from <https://vb-audio.com/Cable/>
   and reboot if it asks. This driver provides the new microphone device.
2. Run **`VibeVoice.exe`**. It's a single file with nothing to install, and
   Python isn't needed.
3. In **Setup**, make sure *Virtual mic output* is set to **CABLE Input**.
4. In Discord, Zoom, a game, etc., choose **CABLE Output (VB-Audio Virtual Cable)**
   as your microphone.

Windows SmartScreen may warn about an unsigned app the first time. Click
*More info* → *Run anyway*.

## Features

- **23 built-in voices**: Girl, Woman, Old Man, Robot, Mega Echo, Deep,
  Chipmunk, Child, Giant, Demon, Villain, Cyborg, Android, Alien, Radio,
  Telephone, Megaphone, 8-Bit, Cave, Stadium Announcer, Ghost, Underwater,
  plus Clean.
- **Natural female voices**: Girl and Woman learn your average speaking pitch
  and move it to a female range (about 270 Hz and 210 Hz). Your natural
  intonation stays the same, so low and high voices both land in the right
  place. Formants are raised separately (+30% and +19%), which keeps them from
  sounding like a chipmunk. The status bar shows the pitch it has learned for
  you.
- **Pitch and timbre controls**: *Pitch* sets how high or low you sound,
  *Formant (timbre)* sets how big your throat sounds, and *Auto pitch target*
  aims your voice at a fixed pitch. When the target is on, Pitch adds an
  offset on top of it.
- **Monotone robot** replaces your voice's pitch with a single fixed note, and
  you can set that note with *Robot pitch*. **Tremble** and **Breathiness**
  give the Old Man his shaky, airy voice.
- **Test your voice**: record 5 seconds, then play it back through any voice,
  or play the original to compare. You can keep switching voices on the same clip.
- **Hear Myself**: live monitoring through your headphones.
- **Customize**: 26 controls. Use *Save as new* to keep your own voices. Edits
  to custom voices save automatically.
- **Background layers** (Layers tab): stack chaotic backgrounds on top of any
  voice. They play into your mic nonstop, with random timing, so they never loop
  the same way twice.
  - **Zoo**: monkeys, elephants, lions, ducks, sheep, cows, parrots and dogs
  - **Sea**: crashing waves, seagulls, boat horns and a bell buoy
  - **Storm**: heavy rain, gusting wind and rolling thunder
  - **City Traffic**, **Jungle**, **Campfire**, **Construction** and **Space Battle**

  Each layer has its own volume, and there's a master volume. `Ctrl+Alt+L`
  pauses all layers from any app. Layers keep playing when VOICE is off, and
  Mute silences everything. Every sound is synthesized in code, with no
  recordings, so the app stays small.
- **Share voices with a code**: click *Share* to copy a short code for the
  current voice, such as `VV1-69YK7sSXEpbBuJbcpo3UuNhJJtMaeWXEv5E1uiU`. A
  friend clicks *Import code* and pastes it to get the exact same voice. Codes
  are safe to paste in Discord, and a checksum catches typos.
- **Global hotkeys** that work while you're in a game:
  `Ctrl+Alt+1..9` switch voice, `Ctrl+Alt+0` turns the effect on/off,
  `Ctrl+Alt+M` mutes, `Ctrl+Alt+L` pauses background layers.
- Noise gate, level meters, and a monitor volume control.

## Why it's light

- The exe is a single 25 MB file and uses about 70 MB of RAM. Its only
  libraries are numpy and sounddevice, and the UI uses tkinter.
- All DSP is vectorised per audio block, with no per-sample Python loops.
  Effects that are switched off cost nothing. A typical load is **~1% of one
  CPU core**. The heaviest voices (the female voices use a phase-vocoder
  pitch shifter) take about 5%.
- About 50 ms end-to-end latency on WASAPI. Voices that move pitch and timbre
  separately add about 43 ms. Setup → Latency trades latency for stability.

## Troubleshooting

- **Robotic crackles or glitches**: set Latency to *Safe (1024)*.
- **A device won't open**: try the *MME* audio driver in Setup.
- **Background noise comes through**: raise the Noise gate.
- **Echo or feedback while monitoring**: use headphones, not speakers.
- **Girl/Woman too high or too low for you**: change *Auto pitch target*
  (or *Pitch* as a fine offset), then *Save as new*.

Settings and custom voices are stored in `%APPDATA%\VibeVoice\settings.json`.

## Run from source / build the exe

```
pip install -r requirements.txt
python main.py          # run from source
python build.py         # build dist\VibeVoice.exe (needs Python 3.10+)
```

`build.py` builds in a clean virtual environment, so only the app's own
dependencies end up in the exe.

## Project layout

```
main.py               entry point
build.py              builds dist/VibeVoice.exe with PyInstaller
vibevoice/dsp.py      effects: pitch shifters, phase vocoder, reverb, echo, EQ, ...
vibevoice/profiles.py parameter schema + built-in voices
vibevoice/sharecode.py voice <-> share code (packed, CRC-checked, base58)
vibevoice/ambience.py background layers: sound synthesis + real-time mixer
vibevoice/engine.py   audio I/O: mic -> effects -> virtual cable / monitor
vibevoice/hotkeys.py  global hotkeys (Win32 RegisterHotKey)
vibevoice/app.py      GUI
assets/               app icon (generated by tools/make_icon.py)
```

## License

[MIT](LICENSE): free to use, modify and share.
