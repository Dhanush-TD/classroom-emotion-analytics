@echo off
REM Quick start script for Education Emotion Analytics Dashboard
REM Run this from the emotion_project directory

echo.
echo ================================================
echo Education Emotion Analytics Dashboard
echo ================================================
echo.

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo Activating virtual environment...
    call fer_env\Scripts\activate.bat
    if errorlevel 1 (
        echo ERROR: Failed to activate virtual environment
        pause
        exit /b 1
    )
)

echo Virtual environment: %VIRTUAL_ENV%
echo.

REM Navigate to dashboard directory
cd analytics\dashboard
if errorlevel 1 (
    echo ERROR: Could not navigate to dashboard directory
    pause
    exit /b 1
)

echo.
echo Starting Flask server...
echo.
echo ✓ Dashboard will be available at: http://127.0.0.1:5000
echo ✓ Press Ctrl+C to stop the server
echo.

REM Start the Flask app
python app.py

if errorlevel 1 (
    echo.
    echo ERROR: Failed to start Flask app
    echo Troubleshooting:
    echo   1. Check virtual environment is activated
    echo   2. Verify all dependencies are installed
    echo   3. Check if port 5000 is available
    pause
    exit /b 1
)
