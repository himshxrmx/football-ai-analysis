@echo off
REM One-time setup for Windows: creates the .venv folder and installs everything.
REM Needs an internet connection; downloads about 1 GB of packages.
setlocal
cd /d "%~dp0"

set "PYEXE="
for %%V in (3.13 3.12 3.11) do (
    if not defined PYEXE (
        py -%%V -c "import sys" >nul 2>nul && set "PYEXE=py -%%V"
    )
)
if not defined PYEXE (
    python -c "import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)" >nul 2>nul && set "PYEXE=python"
)
if not defined PYEXE (
    echo Python 3.11, 3.12 or 3.13 was not found.
    echo Install Python 3.13 from https://www.python.org/downloads/ and tick "Add python.exe to PATH",
    echo then run setup.bat again.
    pause
    exit /b 1
)
echo Using %PYEXE%

if not exist ".venv\Scripts\python.exe" (
    echo Creating the virtual environment in .venv ...
    %PYEXE% -m venv .venv || goto :error
)
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :error
echo Installing PyTorch (CPU version) ...
".venv\Scripts\python.exe" -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu || goto :error
echo Installing the other packages ...
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
echo Downloading the sample video ...
".venv\Scripts\python.exe" scripts\download_sample.py

echo.
echo Setup finished. Start the app by double-clicking run_app.bat
pause
exit /b 0

:error
echo.
echo Setup stopped because of the error above. See "Troubleshooting" in README.md.
pause
exit /b 1
