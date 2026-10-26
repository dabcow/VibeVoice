"""Build dist/VibeVoice.exe (single file, no console window).

    python build.py

Builds inside a fresh, minimal virtual environment (numpy + sounddevice +
PyInstaller only) so nothing else installed on this machine ends up in the exe.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(ROOT, "build")
VENV = os.path.join(BUILD, "venv")
PY = os.path.join(VENV, "Scripts", "python.exe")
ICON = os.path.join(ROOT, "assets", "icon.ico")
VERSION = (1, 1, 0)

VERSION_INFO = f"""\
VSVersionInfo(
  ffi=FixedFileInfo(filevers={VERSION + (0,)}, prodvers={VERSION + (0,)}),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('ProductName', 'VibeVoice'),
      StringStruct('FileDescription', 'VibeVoice voice changer'),
      StringStruct('FileVersion', '{".".join(map(str, VERSION))}'),
      StringStruct('ProductVersion', '{".".join(map(str, VERSION))}'),
      StringStruct('OriginalFilename', 'VibeVoice.exe'),
      StringStruct('LegalCopyright', 'MIT License')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""


def run(*cmd):
    print(">", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    os.makedirs(BUILD, exist_ok=True)
    if not os.path.exists(PY):
        run(sys.executable, "-m", "venv", VENV)
    run(PY, "-m", "pip", "install", "-q", "--disable-pip-version-check",
        "-r", "requirements.txt", "pyinstaller")
    if not os.path.exists(ICON):
        run(PY, os.path.join("tools", "make_icon.py"))
    version_file = os.path.join(BUILD, "version_info.txt")
    with open(version_file, "w", encoding="utf-8") as f:
        f.write(VERSION_INFO)
    run(PY, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile", "--windowed",
        "--name", "VibeVoice",
        "--icon", ICON,
        "--add-data", ICON + os.pathsep + "assets",
        "--version-file", version_file,
        "--workpath", os.path.join(BUILD, "pyinstaller"),
        "--specpath", BUILD,
        "--distpath", os.path.join(ROOT, "dist"),
        os.path.join(ROOT, "main.py"))
    exe = os.path.join(ROOT, "dist", "VibeVoice.exe")
    print(f"\nBuilt {exe} ({os.path.getsize(exe) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
