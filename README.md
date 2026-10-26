# VibeVoice

A free, lightweight real-time voice changer for Windows. Pick a voice and talk:
your changed voice goes to a virtual microphone that Discord, Zoom, OBS and
games can use.

## Quick start

1. Install **VB-Audio Virtual Cable** (free) from <https://vb-audio.com/Cable/>
   and reboot if it asks. This driver provides the new microphone device.
2. Double-click **`VibeVoice.bat`**. The first run sets everything up
   automatically (Python 3.10+ is required).
3. In **Setup**, make sure *Virtual mic output* is set to **CABLE Input**.
4. In Discord, Zoom, a game, etc., choose **CABLE Output (VB-Audio Virtual Cable)**
   as your microphone.

## Features

- **23 built-in voices**: Girl, Woman, Old Man, Robot, Mega Echo, Deep,
  Chipmunk, Child, Giant, Demon, Villain, Cyborg, Android, Alien, Radio,
  Telephone, Megaphone, 8-Bit, Cave, Stadium Announcer, Ghost, Underwater,
  plus Clean.
- **Formant control**: *Pitch* sets how high or low you sound, and
  *Formant (timbre)* sets how big your throat sounds. Girl and Woman raise the
  pitch more than the formants, so they sound female instead of like a chipmunk.
- **Monotone robot** replaces your voice's pitch with a single fixed note, and
  you can set that note with *Robot pitch*. **Tremble** and **Breathiness**
  give the Old Man his shaky, airy voice.
- **Test your voice**: record 5 seconds, then play it back through any voice,
  or play the original to compare. You can keep switching voices on the same clip.
- **Hear Myself**: live monitoring through your headphones.
- **Customize**: 20 controls (pitch, ring mod, distortion, bit crush,
  chorus/flanger, echo, reverb, EQ, gain). Use *Save as new* to keep your own
  voices. Edits to custom voices save automatically.
- **Global hotkeys** that work while you're in a game:
  `Ctrl+Alt+1..9` switch voice, `Ctrl+Alt+0` turns the effect on/off,
  `Ctrl+Alt+M` mutes.
- Noise gate, level meters, and a monitor volume control.

## Why it's light

- 3 small dependencies (numpy, scipy, sounddevice). The UI uses tkinter, which
  ships with Python.
- All DSP is vectorised per audio block, with no per-sample Python loops.
  Effects that are switched off cost nothing. A typical load is **~1% of one
  CPU core**, and the heaviest voice is still under 5%.
- About 50 ms end-to-end latency on WASAPI. Setup → Latency trades latency
  for stability.

## Troubleshooting

- **Robotic crackles or glitches**: set Latency to *Safe (1024)*.
- **A device won't open**: try the *MME* audio driver in Setup.
- **Background noise comes through**: raise the Noise gate.
- **Echo or feedback while monitoring**: use headphones, not speakers.

Settings and custom voices are stored in `%APPDATA%\VibeVoice\settings.json`.

## Project layout

```
main.py               entry point
VibeVoice.bat         one-click setup + launch
vibevoice/dsp.py      effects: pitch shifter, ring mod, reverb, echo, EQ, ...
vibevoice/profiles.py parameter schema + built-in voices
vibevoice/engine.py   audio I/O: mic -> effects -> virtual cable / monitor
vibevoice/hotkeys.py  global hotkeys (Win32 RegisterHotKey)
vibevoice/app.py      GUI
```
