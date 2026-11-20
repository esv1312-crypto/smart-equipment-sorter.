@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\pyinstaller.exe" (
  echo Сначала запустите install.bat
  pause
  exit /b 1
)
".venv\Scripts\pyinstaller.exe" --noconfirm --clean --onedir --windowed --name SmartEquipmentSorter --collect-all transformers --collect-all torch app.py
echo Сборка завершена: dist\SmartEquipmentSorter
pause
