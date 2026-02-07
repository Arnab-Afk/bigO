@echo off
echo Starting CCP ML API Server...
echo.
echo Server will be available at: http://localhost:8000
echo API documentation at: http://localhost:8000/docs
echo.

cd /d "%~dp0"
python api.py

pause
