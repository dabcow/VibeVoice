"""System-wide hotkeys via Win32 RegisterHotKey (no extra dependencies)."""
import ctypes
import ctypes.wintypes as wt
import queue
import sys
import threading

MOD_ALT, MOD_CONTROL, MOD_NOREPEAT = 0x1, 0x2, 0x4000
WM_HOTKEY, WM_QUIT = 0x0312, 0x0012


class GlobalHotkeys:
    """Registers (modifiers, virtual_key, tag) bindings; fired tags land in `.events`."""

    def __init__(self, bindings):
        self.events = queue.Queue()
        self._tid = None
        if sys.platform == "win32":
            threading.Thread(target=self._run, args=(bindings,), daemon=True).start()

    def _run(self, bindings):
        user32 = ctypes.windll.user32
        self._tid = ctypes.windll.kernel32.GetCurrentThreadId()
        ids = {}
        for i, (mods, vk, tag) in enumerate(bindings, 1):
            if user32.RegisterHotKey(None, i, mods | MOD_NOREPEAT, vk):
                ids[i] = tag
        msg = wt.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY and msg.wParam in ids:
                self.events.put(ids[msg.wParam])
        for i in ids:
            user32.UnregisterHotKey(None, i)

    def stop(self):
        if self._tid:
            ctypes.windll.user32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)
