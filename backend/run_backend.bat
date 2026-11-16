@echo off
cd /d "%~dp0"
echo ====================================================
echo Starting PromptLab Backend (FastAPI + Agentic AI)...
echo URL: http://127.0.0.1:8000
echo Docs: http://127.0.0.1:8000/docs
echo ====================================================
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
pause
