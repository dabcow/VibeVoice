@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo First run: setting up VibeVoice...
  python -m venv .venv || (echo Python 3.10+ is required: https://www.python.org/downloads/ & pause & exit /b 1)
  ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt || (pause & exit /b 1)
)
start "" ".venv\Scripts\pythonw.exe" main.py
