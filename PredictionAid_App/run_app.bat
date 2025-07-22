@echo off
echo Starting PredictionAID App...
echo.

REM Use the working conda environment path
set "CONDA_PYTHON=C:\Users\sdv.werkstudent\.conda\envs\tool_app\python.exe"
if exist "%CONDA_PYTHON%" (
    echo Using conda environment: tool_app
    echo Python path: %CONDA_PYTHON%
    "%CONDA_PYTHON%" run_app.py
    goto :end
)

REM If conda environment not found, show error and try system Python
echo Warning: Conda environment 'tool_app' not found at expected location
echo Expected: %CONDA_PYTHON%
echo.
echo Trying system Python as fallback...

python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python or activate your conda environment manually:
    echo   conda activate tool_app
    echo   cd Tool_App
    echo   python -m streamlit run Main.py
    pause
    exit /b 1
)

echo Using system Python (may not work without proper dependencies)
python run_app.py

:end
pause
