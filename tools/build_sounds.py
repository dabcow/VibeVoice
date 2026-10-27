"""Download and prepare the background-layer recordings.

    python tools/build_sounds.py [download_dir]

Fetches the public-domain (CC0) Freesound previews listed in SOURCES into
download_dir (default: build/sound_downloads), then trims, splits, loops,
levels and encodes them into assets/sounds/ with manifest.json and CREDITS.md.
Development tool only: needs numpy, scipy and soundfile.
"""
import json
import os
import sys
import urllib.request

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 48000

# layer -> [(freesound id, user id, author, title, max bytes to fetch or None)]
SOURCES = {
    "smoke": [
        (819808, 16739153, "Bloofrzo", "Smoke Detector Chirp 1", None),
        (819807, 16739153, "Bloofrzo", "Smoke Detector Chirp 2", None),
        (419239, 3798635, "Techienanna", "Smoke alarm battery low", None),
        (369848, 1480854, "SpliceSound", "Smoke detector alarm, close perspective", None),
    ],
    "firealarm": [
        (319413, 336074, "pushkin", "Fire alarm", None),
        (210513, 1088850, "nigelcoop", "Fire Alarm", None),
        (82797, 1254612, "payattention", "fire alarm", None),
        (325367, 5403671, "MoonlightShadow", "fire house alarm", None),
    ],
    "honking": [
        (451523, 612689, "kyles", "traffic medium jam gridlock throaty angry horn honking", 2500000),
        (182474, 3153523, "keweldog", "car horn", None),
        (434878, 3388578, "MicktheMicGuy", "Car Honking", None),
        (461679, 9679617, "15HPanska_Ruttner_Jan", "05 Horn", None),
        (423990, 4023430, "AmishRob", "Car horn beep beep two beeps honk honk", None),
        (385283, 6394480, "Mullumbimby", "Horn_Doppler_01", None),
        (32417, 276701, "KRAFTWERK2K1", "car_horn", None),
        (349922, 2866779, "DeVern", "Car Horn Honk", None),
        (235363, 3915528, "3863", "car horn 1", None),
        (571348, 12894507, "99021905683", "car horn honking", None),
    ],
    "fastfood": [
        (271443, 672090, "lauramellis", "Fast Food Restaurant", 3000000),
        (455769, 612689, "kyles", "fast food restaurant ordering counter calling customers", 2500000),
        (405148, 3694371, "logancircle2", "Fast Food restaurant", None),
        (844265, 12024110, "chungus43A", "Deep fryer beep noise", None),
    ],
    "lecture": [
        (168503, 3033586, "deleted_user_3033586", "Psychology lecture (Troy Dvorak), October 9 2012, part 1", 6000000),
        (62607, 791247, "bsumusictech", "class_chatter (computer music lecture)", 3000000),
        (425671, 8477079, "bparker4336", "Lecture", None),
        (869630, 19231455, "mandoguy24", "Lecture hall before lecture", None),
        (151217, 140737, "OwlStorm", "Cough", None),
        (208761, 3367616, "Harris85", "Cough", None),
    ],
    "india": [
        (263636, 4536481, "Alcappuccino", "India Streets New Delhi City (Traffic, Tuktuks, Horns)", 3000000),
        (450730, 1147184, "csaszi", "Kolkata market, people selling cloths", None),
        (453005, 1441734, "Lenguaverde", "Mumbai - Market", None),
        (263635, 4536481, "Alcappuccino", "India Trivandrum Street Market", None),
        (585570, 12833693, "Athul_PR", "Indian Street traffic Ambience 2", None),
    ],
    "zoo": [
        (263569, 4536481, "Alcappuccino", "London Battersea Park Zoo", 2000000),
        (103117, 1745410, "geronimo83", "Elephant (Manyara, Tanzania)", 2000000),
        (527845, 11431915, "D.jones", "Elephant Trumpets Growls", None),
        (69570, 706955, "Bidone", "Lion loud", None),
        (69572, 706955, "Bidone", "Lion roaring", None),
        (69571, 706955, "Bidone", "Lions two", None),
        (67361, 706955, "Bidone", "Affen schreit (chimpanzee, Leipzig Zoo)", None),
        (67355, 706955, "Bidone", "Gibbon Monkey", None),
        (362283, 676927, "zachrau", "Sheep bleating", None),
        (210511, 770707, "Yuval", "sheep bleat outdoors", None),
        (177253, 276010, "JosephSardin", "Cow moos", None),
        (163727, 1661766, "felix.blume", "Cow mooing in south of France", None),
        (607226, 887696, "D4XX", "Ducks", None),
        (418509, 5632532, "Mikes-MultiMedia", "single quack from a duck", None),
        (187928, 3488149, "castle81", "Parrots", None),
    ],
    "sea": [
        (412308, 7707368, "straget", "Big waves hit land", 2500000),
        (531015, 9818404, "Noted451", "Ocean Waves", None),
        (353416, 1149179, "squashy555", "Seagull on beach", None),
        (462462, 2752236, "Soojay", "Seagulls clean wildtrack", None),
        (64601, 782688, "Grotelue", "Ship Horn (Hamburg harbour)", None),
        (441223, 6072659, "devy32", "Boat Horn", None),
        (208714, 1756543, "monotraum", "ship horn", None),
    ],
    "storm": [
        (507902, 9004329, "inuetc", "Heavy Rain Sound", None),
        (200272, 2322692, "jmbphilmes", "Rain heavy 2 (rural)", None),
        (446753, 1790434, "BlueDelta", "Heavy Thunder Strike - no Rain", None),
        (194364, 1985044, "Dave Welsh", "Thunder Clap", None),
        (244053, 2971247, "lennyboy", "Thunder", None),
        (97397, 1615488, "jpnien", "Wind strong", None),
    ],
    "city": [
        (36734, 43834, "moxobna", "citystreet3", 2500000),
        (52906, 101023, "vlammenos", "police siren", None),
        (658724, 11755334, "tlabrant1", "Sirens", None),
        (337101, 3474310, "ivolipa", "Group of Dogs Barking", None),
        (163459, 2965892, "LittleBigSounds", "Dog Small Alert Bark", None),
        (182474, 3153523, "keweldog", "car horn", None),
    ],
    "jungle": [
        (424763, 3158178, "jaime_enrique", "Jungle_Choco 01 (Colombia)", None),
        (123087, 1643373, "itsmrjack", "jungle_1 (Palenque, Mexico)", None),
        (123044, 1643373, "itsmrjack", "howler_monkey_jungle_4", None),
        (232748, 3275399, "franciscoguerrero", "Monos Aulladores - Howler monkeys", None),
        (32655, 189568, "greysound", "Frogs and Crickets", None),
        (536759, 1415754, "egomassive", "Frog", None),
    ],
    "campfire": [
        (62487, 43834, "moxobna", "campfire", None),
        (40699, 25133, "Spandau", "campfire", 2500000),
        (522299, 9084007, "Defelozedd94", "Crickets At Night", 2000000),
        (465697, 9159316, "Breviceps", "Owl Hoot", None),
        (398734, 5923045, "Anthousai", "owl", None),
        (151241, 1050391, "kvgarlic", "Owls Forest", 2000000),
    ],
    "construction": [
        (246171, 1733081, "sbda", "Construction Soundscape", None),
        (118042, 2110106, "G.Lamont", "Construction Site", None),
        (273697, 5208258, "thomaspettigrew", "Jack Hammer breaking up concrete", None),
        (42097, 317797, "fauxpress", "Jackhammer", None),
        (271662, 5158123, "HonorHunter", "Jackhammer close", None),
        (567294, 97550, "TRP", "Reversing truck beeps, alley", None),
        (207782, 2046066, "Dvideoguy", "Hammering", None),
        (17012, 57789, "cognito perceptu", "hammering 2", None),
    ],
    "space": [
        (130883, 16058, "Karma-Ron", "low ship humming", None),
        (187119, 1038806, "unfa", "Laser Weapon Sounds", None),
        (214990, 2737063, "peridactyloptrix", "Laser blast x3", None),
        (330293, 4361321, "humanoide9000", "Sci-fi gun shot x6", None),
        (235968, 4265427, "tommccann", "Explosion_01", None),
        (97896, 866470, "Ohrwurm", "Explosion at a construction site", None),
        (206141, 3863800, "radio_illuminati", "Nuclear Alarm", None),
        (165504, 2698284, "ryanconway", "Missile Lock Detected", None),
        (260103, 1089045, "notKlaatu", "Retro Sci Fi Computer", None),
    ],
}

# role per sound: ("bed", seconds, gain) | ("split", events/min, (glo, ghi), max_variants, max_len)
# | ("chunks", events/min, (glo, ghi), n_chunks, chunk_len) | ("whole", events/min, (glo, ghi), max_len)
SPEC = {
    "smoke": {819808: ("whole", 6, (0.5, 0.8), 3), 819807: ("whole", 5, (0.3, 0.5), 3),
              419239: ("split", 5, (0.3, 0.7), 6, 3), 369848: ("whole", 0.8, (0.3, 0.5), 8)},
    "firealarm": {319413: ("bed", 32, 1.0), 210513: ("whole", 1.5, (0.3, 0.5), 9),
                  325367: ("whole", 1.2, (0.3, 0.5), 10), 82797: ("whole", 1.0, (0.5, 0.7), 28)},
    "honking": {451523: ("bed", 75, 1.0), 182474: ("split", 5, (0.3, 0.6), 3, 4),
                434878: ("whole", 4, (0.3, 0.6), 2), 461679: ("whole", 4, (0.3, 0.6), 4),
                423990: ("whole", 4, (0.3, 0.6), 2), 385283: ("whole", 2, (0.3, 0.5), 10),
                32417: ("whole", 4, (0.3, 0.6), 2), 349922: ("whole", 4, (0.3, 0.6), 4),
                235363: ("whole", 3, (0.3, 0.5), 9), 571348: ("whole", 4, (0.3, 0.6), 2)},
    "fastfood": {271443: ("bed", 75, 1.0), 455769: ("bed", 45, 0.5),
                 405148: ("chunks", 2, (0.4, 0.6), 3, 10), 844265: ("whole", 3, (0.15, 0.3), 3)},
    "lecture": {168503: ("bed", 120, 1.0), 869630: ("bed", 40, 0.25),
                151217: ("whole", 2, (0.15, 0.3), 2), 208761: ("split", 1.5, (0.15, 0.3), 3, 3)},
    "india": {263636: ("bed", 75, 1.0), 450730: ("bed", 45, 0.7),
              453005: ("chunks", 3, (0.4, 0.7), 4, 8), 263635: ("chunks", 3, (0.4, 0.7), 5, 8),
              585570: ("chunks", 2, (0.4, 0.6), 4, 8)},
    "zoo": {263569: ("bed", 75, 1.0), 103117: ("split", 4, (0.4, 0.7), 4, 6),
            527845: ("whole", 3, (0.4, 0.7), 3), 69570: ("split", 3, (0.4, 0.7), 4, 6),
            69572: ("split", 3, (0.4, 0.7), 3, 6), 69571: ("split", 2, (0.4, 0.7), 2, 6),
            67361: ("whole", 4, (0.4, 0.7), 4), 67355: ("whole", 3, (0.4, 0.7), 8),
            362283: ("split", 4, (0.3, 0.6), 4, 4), 210511: ("split", 3, (0.3, 0.6), 4, 4),
            177253: ("split", 3, (0.4, 0.7), 4, 4), 163727: ("split", 3, (0.4, 0.7), 3, 4),
            607226: ("whole", 3, (0.3, 0.6), 4), 418509: ("whole", 3, (0.3, 0.5), 2),
            187928: ("whole", 3, (0.3, 0.6), 8)},
    "sea": {412308: ("bed", 75, 1.0), 531015: ("bed", 60, 0.6),
            353416: ("split", 5, (0.3, 0.6), 5, 5), 462462: ("split", 5, (0.3, 0.6), 5, 5),
            64601: ("chunks", 1, (0.4, 0.7), 2, 12), 441223: ("chunks", 1, (0.4, 0.7), 2, 10),
            208714: ("chunks", 1, (0.4, 0.6), 2, 12)},
    "storm": {507902: ("bed", 54, 1.0), 200272: ("bed", 60, 0.6), 97397: ("bed", 32, 0.5),
              446753: ("split", 2, (0.6, 0.9), 3, 15), 194364: ("split", 2, (0.6, 0.9), 3, 15),
              244053: ("whole", 1.5, (0.5, 0.8), 15)},
    "city": {36734: ("bed", 90, 1.0), 52906: ("chunks", 1, (0.3, 0.5), 2, 10),
             658724: ("whole", 1, (0.3, 0.5), 19), 337101: ("chunks", 2, (0.3, 0.5), 2, 6),
             163459: ("whole", 2, (0.2, 0.4), 2), 182474: ("split", 4, (0.3, 0.5), 3, 4)},
    "jungle": {424763: ("bed", 60, 1.0), 123087: ("bed", 75, 0.7),
               123044: ("chunks", 3, (0.4, 0.7), 4, 8), 232748: ("chunks", 3, (0.4, 0.7), 4, 8),
               32655: ("chunks", 2, (0.3, 0.5), 3, 10), 536759: ("whole", 6, (0.3, 0.6), 1)},
    "campfire": {62487: ("bed", 80, 1.0), 522299: ("bed", 60, 0.35),
                 465697: ("whole", 1.5, (0.3, 0.5), 5), 398734: ("whole", 1.5, (0.3, 0.5), 4),
                 151241: ("split", 1, (0.3, 0.5), 3, 6)},
    "construction": {246171: ("bed", 59, 1.0), 118042: ("bed", 35, 0.5),
                     273697: ("whole", 3, (0.4, 0.7), 10), 42097: ("whole", 3, (0.4, 0.7), 10),
                     271662: ("chunks", 2, (0.4, 0.7), 2, 10), 567294: ("chunks", 2, (0.3, 0.5), 3, 8),
                     207782: ("whole", 4, (0.4, 0.7), 5), 17012: ("whole", 4, (0.4, 0.7), 2)},
    "space": {130883: ("bed", 27, 1.0), 187119: ("split", 14, (0.3, 0.6), 10, 2),
              214990: ("whole", 5, (0.3, 0.6), 6), 330293: ("split", 6, (0.3, 0.6), 6, 2),
              235968: ("whole", 3, (0.5, 0.8), 8), 97896: ("whole", 2, (0.5, 0.8), 15),
              206141: ("chunks", 1, (0.25, 0.4), 2, 8), 165504: ("whole", 1, (0.4, 0.6), 18),
              260103: ("whole", 3, (0.3, 0.5), 6)},
}


def load(path):
    x, sr = sf.read(path, dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = np.gcd(SR, sr)
        x = resample_poly(x, SR // g, sr // g)
    # remove DC / rumble below 30 Hz (FFT high-pass)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    X[f < 30] = 0
    return np.fft.irfft(X, len(x))


def fade(x, a=0.01, r=0.05):
    a, r = min(int(a * SR), len(x) // 2), min(int(r * SR), len(x) // 2)
    x = x.copy()
    if a:
        x[:a] *= np.linspace(0, 1, a)
    if r:
        x[-r:] *= np.linspace(1, 0, r)
    return x


def env_db(x, frame=0.02):
    n = int(frame * SR)
    k = len(x) // n
    e = np.sqrt(np.mean(x[:k * n].reshape(k, n) ** 2, axis=1) + 1e-12)
    return 20 * np.log10(e), n


def trim(x, rel_db=-40):
    db, n = env_db(x)
    on = np.flatnonzero(db > db.max() + rel_db)
    if len(on) == 0:
        return x
    s, e = max(0, on[0] * n - int(0.03 * SR)), min(len(x), (on[-1] + 1) * n + int(0.15 * SR))
    return x[s:e]


def split(x, max_variants, max_len):
    """Find separate calls/honks by energy; return the strongest segments."""
    db, n = env_db(x)
    thr = max(db.max() - 22, np.percentile(db, 30) + 8)
    act = db > thr
    segs, i = [], 0
    while i < len(act):
        if act[i]:
            j = i
            gap = 0
            while j < len(act) and gap < int(0.35 / 0.02):
                gap = 0 if act[j] else gap + 1
                j += 1
            segs.append((i, j - gap))
            i = j
        else:
            i += 1
    out = []
    for s, e in segs:
        a, b = max(0, s * n - int(0.05 * SR)), min(len(x), e * n + int(0.25 * SR))
        if (b - a) / SR < 0.15:
            continue
        b = min(b, a + int(max_len * SR))
        out.append(x[a:b])
    out.sort(key=lambda s: -float(np.max(np.abs(s))))
    return [fade(s) for s in out[:max_variants]] or [fade(trim(x)[:int(max_len * SR)])]


def chunks(x, k, length):
    L = int(length * SR)
    if len(x) <= L:
        return [fade(x, 0.3, 0.6)]
    starts = np.linspace(0, len(x) - L, k + 2)[1:-1].astype(int)
    return [fade(x[s:s + L], 0.3, 0.6) for s in starts]


def level(x, target=0.1, win=0.5, smooth=3.0):
    """Slow automatic leveling (+-12 dB) plus a soft 14 dB-crest peak limit, so a
    loop sounds equally loud wherever playback happens to be in it."""
    n = int(win * SR)
    k = max(1, len(x) // n)
    rms = np.sqrt(np.mean(x[:k * n].reshape(k, n) ** 2, axis=1) + 1e-12)
    w = max(1, int(smooth / win))
    rms = np.convolve(np.pad(rms, w, mode="edge"), np.ones(2 * w + 1) / (2 * w + 1), "valid")
    gain = np.clip(target / rms, 0.25, 4.0)
    g = np.interp(np.arange(len(x)), np.arange(k) * n + n / 2, gain)
    y = x * g
    c = 5 * target
    return c * np.tanh(y / c)


def bed(x, seconds):
    start = min(int(1.0 * SR), len(x) // 10)
    x = level(x[start:start + int((seconds + 2) * SR)])
    f = int(2.0 * SR)  # seamless loop: cross-fade the tail into the head
    body = x[:-f].copy()
    w = np.sin(np.linspace(0, np.pi / 2, f)) ** 2
    body[:f] = x[:f] * w + x[-f:] * (1 - w)
    return body / (np.sqrt(np.mean(body ** 2)) + 1e-12) * 0.1  # -20 dBFS RMS


def peaknorm(x, peak=0.89):
    return x * (peak / (np.max(np.abs(x)) + 1e-12))


def write(path, x):
    # block-wise: libsndfile's Vorbis encoder overflows the stack on long single writes (Windows)
    with sf.SoundFile(path, "w", SR, 1, format="OGG", subtype="VORBIS",
                      compression_level=0.6) as f:
        for i in range(0, len(x), 8192):
            f.write(x[i:i + 8192].astype(np.float32))


def process(DL, OUT):
    manifest, credits, total = {}, [], 0
    for layer, items in SOURCES.items():
        os.makedirs(os.path.join(OUT, layer), exist_ok=True)
        entry = {"beds": [], "events": []}
        for sid, uid, author, title, _cap in items:
            role = SPEC[layer].get(sid)
            if role is None:  # downloaded but not used (e.g. a second lecturer's voice)
                continue
            src = os.path.join(DL, layer, f"{sid}.mp3")
            try:
                x = load(src)
            except Exception as ex:  # noqa: BLE001
                print(f"!! {layer}/{sid}: cannot decode ({ex})")
                continue
            dur = len(x) / SR
            if role[0] == "bed":
                clip = bed(x, role[1])
                name = f"{layer}/bed_{sid}.ogg"
                write(os.path.join(OUT, name), clip)
                entry["beds"].append({"file": name, "gain": role[2]})
                files = [name]
            else:
                if role[0] == "split":
                    parts = split(x, role[3], role[4])
                elif role[0] == "chunks":
                    parts = chunks(x, role[3], role[4])
                else:
                    parts = [fade(trim(x)[:int(role[3] * SR)])]
                files = []
                for i, p in enumerate(parts):
                    name = f"{layer}/{sid}_{i}.ogg"
                    write(os.path.join(OUT, name), peaknorm(p))
                    files.append(name)
                entry["events"].append({"name": title, "files": files, "rate": role[1],
                                        "gain": list(role[2])})
            size = sum(os.path.getsize(os.path.join(OUT, f)) for f in files)
            total += size
            print(f"{layer:12s} {sid:>7} {role[0]:6s} src {dur:6.1f}s -> {len(files)} file(s) "
                  f"{sum(sf.info(os.path.join(OUT, f)).duration for f in files):6.1f}s {size / 1e3:6.0f} kB  {title[:40]}")
            credits.append((layer, sid, author, title))
        manifest[layer] = entry

    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    with open(os.path.join(OUT, "CREDITS.md"), "w", encoding="utf-8") as f:
        f.write("# Sound credits\n\nAll background-layer sounds come from [Freesound](https://freesound.org) "
                "and were released by their authors under **Creative Commons 0** (public domain). "
                "They were trimmed, mixed to mono, looped and level-matched for VibeVoice.\n"
                "Thank you to every recordist below.\n\n| Layer | Sound | Author | Source |\n|---|---|---|---|\n")
        for layer, sid, author, title in credits:
            f.write(f"| {layer} | {title} | {author} | https://freesound.org/s/{sid}/ |\n")
    print(f"\ntotal {total / 1e6:.1f} MB in {sum(len(e['events']) + len(e['beds']) for e in manifest.values())} sources")



def download(dl):
    for layer, items in SOURCES.items():
        os.makedirs(os.path.join(dl, layer), exist_ok=True)
        for sid, uid, _author, _title, cap in items:
            path = os.path.join(dl, layer, f"{sid}.mp3")
            if os.path.exists(path):
                continue
            req = urllib.request.Request(
                f"https://cdn.freesound.org/previews/{sid // 1000}/{sid}_{uid}-hq.mp3",
                headers={"Range": f"bytes=0-{cap}"} if cap else {})
            with urllib.request.urlopen(req) as r, open(path, "wb") as f:
                f.write(r.read())
            print("downloaded", layer, sid)


if __name__ == "__main__":
    dl = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "sound_downloads")
    out = os.path.join(ROOT, "assets", "sounds")
    download(dl)
    os.makedirs(out, exist_ok=True)
    process(dl, out)
