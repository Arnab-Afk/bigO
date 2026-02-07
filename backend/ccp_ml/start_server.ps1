# Start CCP ML API Server
# PowerShell startup script

Write-Host "Starting CCP ML API Server..." -ForegroundColor Green
Write-Host ""
Write-Host "Server will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "API documentation at: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

# Check if uvicorn is available
if (Get-Command uvicorn -ErrorAction SilentlyContinue) {
    Write-Host "Starting with uvicorn (hot reload enabled)..." -ForegroundColor Yellow
    uvicorn api:app --reload --host 0.0.0.0 --port 8000
} else {
    Write-Host "Starting with Python..." -ForegroundColor Yellow
    python api.py
}

Read-Host -Prompt "Press Enter to exit"
