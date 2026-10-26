@echo off
cd /d "%~dp0"
if exist .git (
  where git >nul 2>nul && (echo Checking for updates... & git pull --ff-only || echo Update skipped)
)
python -m pip install -q -r requirements.txt
python -m streamlit run app.py
