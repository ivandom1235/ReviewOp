# ReviewOp Project Runner Script
# This script automates setup for backend and frontend.

$ErrorActionPreference = 'Stop'

$success = 'Green'
$warning = 'Yellow'
$errorColor = 'Red'
$info = 'Cyan'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendPath = Join-Path $root 'backend'
$frontendPath = Join-Path $root 'frontend'
$venvPath = Join-Path $backendPath 'venv'
$activatePath = Join-Path $venvPath 'Scripts\\Activate.ps1'

function Write-Section([string]$text, [string]$color = 'Cyan') {
    Write-Host ''
    Write-Host '========================================' -ForegroundColor $color
    Write-Host $text -ForegroundColor $color
    Write-Host '========================================' -ForegroundColor $color
}

function Assert-CommandAvailable([string]$commandName, [string]$friendlyName, [string]$installHint) {
    if (-not (Get-Command $commandName -ErrorAction SilentlyContinue)) {
        Write-Host "[x] $friendlyName not found. $installHint" -ForegroundColor $errorColor
        exit 1
    }
}

Write-Section 'ReviewOp - Project Runner'

Write-Host 'Checking Python installation...' -ForegroundColor $info
Assert-CommandAvailable -commandName 'python' -friendlyName 'Python' -installHint 'Please install Python 3.10-3.13.'
$pythonVersion = (python --version 2>&1 | Out-String).Trim()
Write-Host "[ok] Python found: $pythonVersion" -ForegroundColor $success

$pythonVersionTuple = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$pythonMajorMinor = [version]$pythonVersionTuple
if ($pythonMajorMinor -lt [version]'3.10' -or $pythonMajorMinor -ge [version]'3.14') {
    Write-Host "[x] Unsupported Python version: $pythonVersionTuple" -ForegroundColor $errorColor
    Write-Host '    Use Python 3.10, 3.11, 3.12, or 3.13 for this project dependencies.' -ForegroundColor $errorColor
    exit 1
}

Write-Host 'Checking Node.js installation...' -ForegroundColor $info
Assert-CommandAvailable -commandName 'node' -friendlyName 'Node.js' -installHint 'Please install Node.js.'
$nodeVersion = (node --version 2>&1 | Out-String).Trim()
Write-Host "[ok] Node.js found: $nodeVersion" -ForegroundColor $success

Write-Section 'Setting up Backend'

if (-not (Test-Path $backendPath)) {
    Write-Host '[x] backend folder not found.' -ForegroundColor $errorColor
    exit 1
}

Push-Location $backendPath
try {
    if (-not (Test-Path $venvPath)) {
        Write-Host 'Creating Python virtual environment...' -ForegroundColor $info
        python -m venv venv
        if ($LASTEXITCODE -ne 0) {
            Write-Host '[x] Failed to create virtual environment.' -ForegroundColor $errorColor
            exit 1
        }
        Write-Host '[ok] Virtual environment created.' -ForegroundColor $success
    } else {
        Write-Host '[ok] Virtual environment already exists.' -ForegroundColor $success
    }

    if (-not (Test-Path $activatePath)) {
        Write-Host '[x] Virtual environment activation script not found.' -ForegroundColor $errorColor
        exit 1
    }

    Write-Host 'Activating virtual environment...' -ForegroundColor $info
    . $activatePath
    Write-Host '[ok] Virtual environment activated.' -ForegroundColor $success

    Write-Host 'Installing Python dependencies...' -ForegroundColor $info
    python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host '[x] Failed to install backend dependencies.' -ForegroundColor $errorColor
        exit 1
    }
    Write-Host '[ok] Backend dependencies installed.' -ForegroundColor $success
}
finally {
    Pop-Location
}

Write-Section 'IMPORTANT: Database Configuration' $warning
Write-Host 'Before running the backend, please update:' -ForegroundColor $warning
Write-Host '  Location: backend\core\config.py' -ForegroundColor $warning
Write-Host '  Update: MySQL username and password' -ForegroundColor $warning
Read-Host 'Press ENTER once you have updated the config file'

Write-Section 'Setting up Frontend'

if (-not (Test-Path $frontendPath)) {
    Write-Host '[x] frontend folder not found.' -ForegroundColor $errorColor
    exit 1
}

Push-Location $frontendPath
try {
    Write-Host 'Installing Node.js dependencies...' -ForegroundColor $info
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host '[x] Failed to install frontend dependencies.' -ForegroundColor $errorColor
        exit 1
    }
    Write-Host '[ok] Frontend dependencies installed.' -ForegroundColor $success
}
finally {
    Pop-Location
}

Write-Section 'Setup Complete!' $success
Write-Host 'To run the project:' -ForegroundColor $info
Write-Host ''
Write-Host 'Option 1: Run in separate terminals (Recommended)' -ForegroundColor $info
Write-Host '  Terminal 1: cd backend; .\venv\Scripts\Activate.ps1; python -m uvicorn app:app --host 127.0.0.1 --port 8000' -ForegroundColor $warning
Write-Host '  Terminal 2: cd frontend; npm run dev' -ForegroundColor $warning
Write-Host ''
Write-Host 'Option 2: Use run-services.ps1 script (experimental)' -ForegroundColor $info
Write-Host '  .\run-services.ps1' -ForegroundColor $warning
Write-Host ''
Write-Host 'After starting services:' -ForegroundColor $info
Write-Host '  - Backend: http://127.0.0.1:8000/docs' -ForegroundColor $warning
Write-Host '  - Frontend: Check console for the dev server URL' -ForegroundColor $warning
