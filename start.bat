@echo off
title Gemini to OpenAI Adapter (Visual Studio Copilot)
cd /d "%~dp0"

echo ================================================================
echo Starting Gemini to OpenAI Adapter for Visual Studio Copilot
echo ================================================================

set PYTHON_CMD=python
if exist "C:\Users\Brilliance\AppData\Local\Python\bin\python.exe" (
    set PYTHON_CMD="C:\Users\Brilliance\AppData\Local\Python\bin\python.exe"
)

%PYTHON_CMD% server.py --host 0.0.0.0 --port 8000 %*
pause
