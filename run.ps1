# Maoz Semantic Search POC Auto-Runner for PowerShell

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "Starting Maoz Semantic Search POC Auto-Runner" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

# Check if Python is installed
try {
    $pythonVersion = & python --version 2>&1
    Write-Host "Using Python version: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Python was not found on your system PATH. Please install Python and try again." -ForegroundColor Red
    Read-Host "Press Enter to exit..."
    exit 1
}

$VENV_DIR = ".venv"

# Create virtual environment if it doesn't exist
if (-not (Test-Path -Path $VENV_DIR)) {
    Write-Host "Creating virtual environment in $VENV_DIR..." -ForegroundColor Yellow
    & python -m venv $VENV_DIR
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create virtual environment." -ForegroundColor Red
        Read-Host "Press Enter to exit..."
        exit 1
    }
    Write-Host "Virtual environment created successfully." -ForegroundColor Green
} else {
    Write-Host "Virtual environment $VENV_DIR already exists." -ForegroundColor Green
}

# Activate the virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
$activateScript = Join-Path $VENV_DIR "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    . $activateScript
} else {
    Write-Host "[ERROR] Could not find activation script: $activateScript" -ForegroundColor Red
    Read-Host "Press Enter to exit..."
    exit 1
}

# Install/Upgrade dependencies
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Yellow
& pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install dependencies." -ForegroundColor Red
    Read-Host "Press Enter to exit..."
    exit 1
}

# Run the Streamlit application
Write-Host "Launching Streamlit application..." -ForegroundColor Yellow
& streamlit run app.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Streamlit exited with code $LASTEXITCODE." -ForegroundColor Orange
    Read-Host "Press Enter to exit..."
}
