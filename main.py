import sys


def selftest(out_path):
    """Load every background layer and mix a few seconds; write a JSON report.

    Used to check a packaged build: VibeVoice.exe --selftest report.json
    """
    import json
    import time

    import numpy as np

    from vibevoice import ambience

    report = {}
    for lid in ambience.LAYER_IDS:
        try:
            t = time.perf_counter()
            bank = ambience.build(lid)
            ambience.BANKS[lid] = bank
            player = ambience.LayerPlayer()
            player.configure({lid: 1.0})
            y = np.concatenate([player.process(512) for _ in range(200)])
            report[lid] = {"ok": bool(np.all(np.isfinite(y)) and np.abs(y).max() > 0),
                           "seconds": round(time.perf_counter() - t, 3),
                           "peak": round(float(np.abs(y).max()), 3)}
            ambience.release(lid)
        except Exception as ex:  # noqa: BLE001 - report every failure
            report[lid] = {"ok": False, "error": repr(ex)}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--selftest":
        selftest(sys.argv[2])
    else:
        from vibevoice.app import main
        main()
