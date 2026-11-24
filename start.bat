@echo off
cd /d "%~dp0"

set VERSION=
if exist "%~dp0VERSION" (
    set /p VERSION=<"%~dp0VERSION"
)
if "%VERSION%"=="" set VERSION=1.4.0

title Copilot OpenAI Adapter v%VERSION% (Visual Studio Copilot)

echo ================================================================
echo Starting Copilot OpenAI Adapter for Visual Studio Copilot v%VERSION%
echo ================================================================

set PYTHON_CMD=python
if exist "C:\Users\Brilliance\AppData\Local\Python\bin\python.exe" (
    set PYTHON_CMD="C:\Users\Brilliance\AppData\Local\Python\bin\python.exe"
)

%PYTHON_CMD% server.py --host 0.0.0.0 --port 8000 %*
pause
