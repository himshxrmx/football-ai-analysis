@echo off
REM Starts the Football Video Analysis app at http://localhost:8501
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo The virtual environment is missing. Run setup.bat first.
    pause
    exit /b 1
)
REM Open the browser a few seconds after the server starts (skip with FOOTBALL_NO_BROWSER=1).
if not defined FOOTBALL_NO_BROWSER start "" /b cmd /c "ping -n 7 127.0.0.1 >nul & start "" http://localhost:8501"
echo Starting the app. It opens in your browser at http://localhost:8501
echo Keep this window open while you use the app. Close it to stop the app.
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 %*
