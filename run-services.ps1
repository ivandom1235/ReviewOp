
# ReviewOp - Run Both Frontend and Backend Services
# This script starts both services in new terminal windows

function Test-LocalPortInUse {
    param(
        [int]$Port
    )
    $conn = Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    return $null -ne $conn
}

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

# Check backend runtime dependencies inside venv
if (-not (Test-Path "backend\venv\Scripts\python.exe")) {
    Write-Host "Error: backend\\venv\\Scripts\\python.exe not found." -ForegroundColor Red
    exit 1
}

Push-Location backend
& .\venv\Scripts\python.exe -c "import uvicorn" 2>$null
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    Write-Host "Error: uvicorn is missing in backend venv. Re-run run-project.ps1." -ForegroundColor Red
    exit 1
}
Pop-Location

# Check frontend dev runtime dependency
if (-not (Test-Path "frontend\node_modules\vite")) {
    Write-Host "Error: vite is missing in frontend/node_modules. Re-run run-project.ps1." -ForegroundColor Red
    exit 1
}

$backendPort = 8000
$frontendPort = 5173
$startedBackend = $false
$startedFrontend = $false

Write-Host "Starting Backend Server..." -ForegroundColor Green
Write-Host "  Location: backend" -ForegroundColor Yellow
Write-Host "  Command: python -m uvicorn app:app --host 127.0.0.1 --port 8000" -ForegroundColor Yellow

if (Test-LocalPortInUse -Port $backendPort) {
    Write-Host "  Skipped: Port $backendPort is already in use (backend likely already running)." -ForegroundColor Yellow
} else {
    # Start backend in new PowerShell window
    $backendScript = @"
cd backend
venv\Scripts\Activate.ps1
Write-Host '[OK] Virtual environment activated' -ForegroundColor Green
Write-Host 'Starting Uvicorn server...' -ForegroundColor Cyan
python -m uvicorn app:app --host 127.0.0.1 --port 8000
"@

    Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendScript
    $startedBackend = $true
}

Write-Host ""
Write-Host "Starting Frontend Development Server..." -ForegroundColor Green
Write-Host "  Location: frontend" -ForegroundColor Yellow
Write-Host "  Command: npm run dev" -ForegroundColor Yellow

if (Test-LocalPortInUse -Port $frontendPort) {
    Write-Host "  Skipped: Port $frontendPort is already in use (frontend likely already running)." -ForegroundColor Yellow
} else {
    # Start frontend in new PowerShell window
    $frontendScript = @"
cd frontend
npm run dev
"@

    Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendScript
    $startedFrontend = $true
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
if ($startedBackend -or $startedFrontend) {
    Write-Host "Services Started!" -ForegroundColor Green
} else {
    Write-Host "No new services started (already running on expected ports)." -ForegroundColor Yellow
}
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Backend will be available at: http://127.0.0.1:8000/docs" -ForegroundColor Yellow
Write-Host "Check the frontend terminal for the dev server URL" -ForegroundColor Yellow
