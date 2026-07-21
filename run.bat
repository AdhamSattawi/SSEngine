@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo Starting Maoz Semantic Search POC Auto-Runner
echo ===================================================

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found on your system PATH.
    echo Please install Python and add it to your environment variables.
    pause
    exit /b 1
)

:: Set virtual environment folder name
set VENV_DIR=.venv

:: Create virtual environment if it doesn't exist
if not exist "%VENV_DIR%" (
    echo Creating virtual environment in %VENV_DIR%...
    python -m venv %VENV_DIR%
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment %VENV_DIR% already exists.
)

:: Activate the virtual environment
echo Activating virtual environment...
call %VENV_DIR%\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

:: Install/Upgrade dependencies
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

:: Run the Streamlit application
echo Launching Streamlit application...
streamlit run app.py

:: Keep window open if streamlit exits with error
if %errorlevel% neq 0 (
    echo [WARNING] Streamlit exited with code %errorlevel%.
    pause
)

deactivate
