@echo off
cd /d "%~dp0"
python -m pip install -q -r requirements.txt
python -m idxbt schedule --time 17:00
echo.
echo Done. Data will now update itself every weekday at 5 PM and at each login.
echo To undo:  python -m idxbt schedule --remove
pause
