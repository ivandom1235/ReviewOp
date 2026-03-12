# ReviewOp - Run Both Frontend and Backend Services
# This script starts both services in new terminal windows

Write-Host "Starting ReviewOp Services..." -ForegroundColor Cyan
Write-Host ""

# Check if backend venv exists
if (-not (Test-Path "backend\venv")) {
    Write-Host "Error: Virtual environment not found. Run run-project.ps1 first." -ForegroundColor Red
    exit 1
}

# Check if node_modules exists in frontend
if (-not (Test-Path "frontend\node_modules")) {
    Write-Host "Error: Frontend dependencies not installed. Run run-project.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host "Starting Backend Server..." -ForegroundColor Green
Write-Host "  Location: backend" -ForegroundColor Yellow
Write-Host "  Command: python -m uvicorn app:app --host 127.0.0.1 --port 8000" -ForegroundColor Yellow

# Start backend in new PowerShell window
$backendScript = @"
cd backend
venv\Scripts\Activate.ps1
Write-Host '[OK] Virtual environment activated' -ForegroundColor Green
Write-Host 'Starting Uvicorn server...' -ForegroundColor Cyan
python -m uvicorn app:app --host 127.0.0.1 --port 8000
"@

Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendScript

Write-Host ""
Write-Host "Starting Frontend Development Server..." -ForegroundColor Green
Write-Host "  Location: frontend" -ForegroundColor Yellow
Write-Host "  Command: npm run dev" -ForegroundColor Yellow

# Start frontend in new PowerShell window
$frontendScript = @"
cd frontend
npm run dev
"@

Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendScript

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Services Started!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Backend will be available at: http://127.0.0.1:8000/docs" -ForegroundColor Yellow
Write-Host "Check the frontend terminal for the dev server URL" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press ENTER to continue..." -ForegroundColor Cyan
Read-Host
