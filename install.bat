@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === Установка зависимостей ===
py -3.10 -m venv .venv
if errorlevel 1 (
  echo Не удалось создать окружение Python 3.10. Установите Python 3.10 x64.
  pause
  exit /b 1
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Ошибка установки. Проверьте интернет и версию Python.
  pause
  exit /b 1
)
echo Установка завершена. Запустите run.bat
pause
