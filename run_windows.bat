@echo off
setlocal
cd /d "%~dp0"

echo ================================================
echo LOOPLAB - SMALL CIRCULAR LOOP ANTENNA
 echo ================================================

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating Python 3.12 virtual environment...
    py -3.12 -m venv .venv
    if errorlevel 1 (
        echo Could not create Python 3.12 environment.
        echo Install Python 3.12 or change py -3.12 in this file to your installed version.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
echo [2/3] Installing requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed. Check your internet connection and Python version.
    pause
    exit /b 1
)

echo [3/3] Starting Streamlit...
python -m streamlit run app.py
pause
