# Quick start script for Education Emotion Analytics Dashboard
# Run this from the emotion_project directory in PowerShell

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "Education Emotion Analytics Dashboard" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment is activated
if (-not (Test-Path env:VIRTUAL_ENV)) {
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & ".\fer_env\Scripts\Activate.ps1"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to activate virtual environment" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host "Virtual environment: $env:VIRTUAL_ENV" -ForegroundColor Green
Write-Host ""

# Navigate to dashboard directory
Set-Location analytics\dashboard
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Could not navigate to dashboard directory" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Starting Flask server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "✓ Dashboard will be available at: http://127.0.0.1:5000" -ForegroundColor Green
Write-Host "✓ Press Ctrl+C to stop the server" -ForegroundColor Green
Write-Host ""

# Start the Flask app
python app.py

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Failed to start Flask app" -ForegroundColor Red
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Check virtual environment is activated" -ForegroundColor Gray
    Write-Host "  2. Verify all dependencies are installed" -ForegroundColor Gray
    Write-Host "  3. Check if port 5000 is available" -ForegroundColor Gray
    Read-Host "Press Enter to exit"
    exit 1
}
