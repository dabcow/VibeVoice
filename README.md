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

- **27 built-in voices**: Girl, Woman, Old Man, Robot, Mega Echo, Deep,
  Chipmunk, Child, Giant, Demon, Villain, Cyborg, Android, Alien, Radio,
  Telephone, Megaphone, 8-Bit, Cave, Stadium Announcer, Ghost, Underwater,
  Right Ear, Circular, Circular Mono, Random Pitch, plus Clean.
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
- **Random Pitch** jumps your voice to a new random pitch every second. Each
  pick is drawn from a bell curve around your normal pitch. The *Random pitch*
  slider (Customize → Pitch) sets its standard deviation: 4 semitones in
  Random Pitch, so most picks land within ±8. The **Voice intensity** slider
  scales that spread: 50% keeps the jumps small, and 150% makes them wild.
  *Random interval* sets the timing (0.2–5 s). Turn up *Random pitch* on any
  voice, such as Woman or Robot, to make it wander too.
- **Test your voice**: record 5 seconds, then play it back through any voice,
  or play the original to compare. You can keep switching voices on the same clip.
- **Hear Myself**: live monitoring through your headphones.
- **Position your voice** (Customize → Position): *Pan* moves your voice
  between the ears, and the **Right Ear** voice puts it in the right channel
  only. **Circular** makes your voice orbit around the listener's head ("8D
  audio"). It uses panning, a tiny delay at the far ear, and muffling behind
  the head, and is best on headphones. Discord voice channels, most game chat
  and normal Zoom calls send the mic in mono, so listeners there hear you
  centred. The full effect comes through in Hear Myself, the voice test, OBS,
  and Zoom's stereo "Original sound" mode. Background layers stay centred.
- **Circular Mono** (or *Mono-safe orbit* in Customize → Position) is the
  orbit for Discord and game chat. Left and right are lost in mono, so it
  moves your voice by distance: close and clear in front, further away and
  muffled behind, with a small Doppler pitch bend (about ±25 cents) as it
  comes and goes. Its two channels always add up to the same signal, so the
  mono mix has none of the phasey sound the headphone version gets there.
  On headphones it still circles left and right.
- **Voice intensity** (Setup tab): one slider scales every voice, from 0% (your
  normal voice) through 100% (the voice as designed) to 150% (exaggerated). It
  turns down the amounts, such as pitch shift, robot, distortion, echo and EQ,
  while keeping each voice's character. Girl at 50% is a gentler Girl, not a
  different voice.
- **Customize**: 31 controls. Use *Save as new* to keep your own voices. Edits
  to custom voices save automatically.
- **Background layers** (Layers tab): stack chaotic backgrounds on top of any
  voice. They play into your mic nonstop, with random timing, so they never loop
  the same way twice. All 14 layers use **real recordings**:
  - **Zoo**: elephants, lions, chimps, gibbons, sheep, cows, ducks and parrots
  - **Sea**: big waves, seagulls and ship horns from a real harbour
  - **Storm**: heavy rain, strong wind and real thunder strikes
  - **City Traffic**, **Jungle**, **Campfire**, **Construction** and **Space Battle**
  - **Smoke Detector**: a low-battery chirp, again and again
  - **Fire Alarm**: a building alarm ringing nonstop, with announcements
  - **Honking**: gridlocked traffic and endless horns
  - **Fast Food Restaurant**: lunch-rush chatter, orders called out, fryer beeps
  - **University Lecture**: a professor lecturing to a full hall
  - **Indian Street Market**: vendors calling out, tuk-tuks and horns

  Each layer has its own volume, and there's a master volume. Each layer's
  **Intensity** slider sets how often its sounds fire and how hard they hit,
  from calm (0.2×) to chaos (3×). For example, Honking goes from about 5 horns
  a minute to over 100. `Ctrl+Alt+L`
  pauses all layers from any app. Layers keep playing when VOICE is off, and
  Mute silences everything. Every layer is level-matched, so switching between
  them doesn't jump in volume.
- **Share voices with a code**: click *Share* to copy a short code for the
  current voice, such as `VV1-69YK7sSXEpbBuJbcpo3UuNhJJtMaeWXEv5E1uiU`. A
  friend clicks *Import code* and pastes it to get the exact same voice. Codes
  are safe to paste in Discord, and a checksum catches typos.
- **Global hotkeys** that work while you're in a game:
  `Ctrl+Alt+1..9` switch voice, `Ctrl+Alt+0` turns the effect on/off,
  `Ctrl+Alt+M` mutes, `Ctrl+Alt+L` pauses background layers.
- Noise gate, level meters, and a monitor volume control.

## Why it's light

- The exe is a single file, including 19 MB of compressed recordings, and uses
  about 70 MB of RAM. Each background layer you switch on adds 1–22 MB and
  loads in under 0.2 s. Its only libraries are numpy, sounddevice and
  soundfile, and the UI uses tkinter.
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
vibevoice/ambience.py background layers: loads recordings + real-time mixer
vibevoice/engine.py   audio I/O: mic -> effects -> virtual cable / monitor
vibevoice/hotkeys.py  global hotkeys (Win32 RegisterHotKey)
vibevoice/app.py      GUI
assets/               app icon (tools/make_icon.py) and layer sounds (tools/build_sounds.py)
```

## Sound credits

The background-layer recordings are public-domain (CC0) sounds from [Freesound](https://freesound.org). The full list of recordists and source links is in [assets/sounds/CREDITS.md](assets/sounds/CREDITS.md). `tools/build_sounds.py` re-downloads and rebuilds them from those sources.

## License

[MIT](LICENSE): free to use, modify and share.
