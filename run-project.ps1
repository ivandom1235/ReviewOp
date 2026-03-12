# ReviewOp Project Runner Script
# This script automates the setup and running of both frontend and backend

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ReviewOp - Project Runner" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Color helpers
$success = "Green"
$warning = "Yellow"
$error_color = "Red"
$info = "Cyan"

# Check if Python 3.13 is installed
Write-Host "Checking Python installation..." -ForegroundColor $info
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor $success
} else {
    Write-Host "✗ Python not found. Please install Python 3.13+" -ForegroundColor $error_color
    exit 1
}

# Check if Node.js is installed
Write-Host "Checking Node.js installation..." -ForegroundColor $info
$nodeVersion = node --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Node.js found: $nodeVersion" -ForegroundColor $success
} else {
    Write-Host "✗ Node.js not found. Please install Node.js" -ForegroundColor $error_color
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setting up Backend" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Navigate to backend
cd backend

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor $info
    python -m venv venv
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Virtual environment created" -ForegroundColor $success
    } else {
        Write-Host "✗ Failed to create virtual environment" -ForegroundColor $error_color
        exit 1
    }
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor $info
& .\venv\Scripts\Activate.ps1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Virtual environment activated" -ForegroundColor $success
}

# Install requirements
Write-Host "Installing Python dependencies..." -ForegroundColor $info
pip install -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Backend dependencies installed" -ForegroundColor $success
} else {
    Write-Host "✗ Failed to install backend dependencies" -ForegroundColor $error_color
    exit 1
}

# Check if config.py needs setup
Write-Host ""
Write-Host "========================================" -ForegroundColor $warning
Write-Host "⚠ IMPORTANT: Database Configuration" -ForegroundColor $warning
Write-Host "========================================" -ForegroundColor $warning
Write-Host "Before running the backend, please update:" -ForegroundColor $warning
Write-Host "  Location: backend\core\config.py" -ForegroundColor $warning
Write-Host "  Update: MySQL username and password" -ForegroundColor $warning
Write-Host ""
Write-Host "Press ENTER once you've updated the config file..."
Read-Host

# Navigation back to root
cd ..

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setting up Frontend" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Navigate to frontend
cd frontend

# Install dependencies
Write-Host "Installing Node.js dependencies..." -ForegroundColor $info
npm install
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Frontend dependencies installed" -ForegroundColor $success
} else {
    Write-Host "✗ Failed to install frontend dependencies" -ForegroundColor $error_color
    exit 1
}

# Navigate back to root
cd ..

Write-Host ""
Write-Host "========================================" -ForegroundColor $success
Write-Host "Setup Complete!" -ForegroundColor $success
Write-Host "========================================" -ForegroundColor $success
Write-Host ""
Write-Host "To run the project:" -ForegroundColor $info
Write-Host ""
Write-Host "Option 1: Run in separate terminals (Recommended)" -ForegroundColor $info
Write-Host "  Terminal 1: cd backend && venv\Scripts\activate && python -m uvicorn app:app --host 127.0.0.1 --port 8000" -ForegroundColor $warning
Write-Host "  Terminal 2: cd frontend && npm run dev" -ForegroundColor $warning
Write-Host ""
Write-Host "Option 2: Use run-services.ps1 script (experimental)" -ForegroundColor $info
Write-Host "  .\run-services.ps1" -ForegroundColor $warning
Write-Host ""
Write-Host "After starting services:" -ForegroundColor $info
Write-Host "  - Backend: http://127.0.0.1:8000/docs" -ForegroundColor $warning
Write-Host "  - Frontend: Check console for the dev server URL" -ForegroundColor $warning
Write-Host ""
