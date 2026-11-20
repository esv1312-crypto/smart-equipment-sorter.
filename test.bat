@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -m py_compile app.py core.py
if errorlevel 1 goto failed
python -m unittest discover -s tests -v
if errorlevel 1 goto failed
echo.
echo All checks passed.
pause
exit /b 0
:failed
echo.
echo Checks failed. See the error above.
pause
exit /b 1
