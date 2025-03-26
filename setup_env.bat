@echo off
setlocal enabledelayedexpansion

REM Set your virtual environment directory
set VENV_DIR=.venv

REM Create the virtual environment
echo Creating virtual environment with Python 3.12.2...
python -m venv %VENV_DIR%

REM Activate the virtual environment
call %VENV_DIR%\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
if exist requirements.txt (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
) else (
    echo No requirements.txt found. Skipping dependency installation.
)

echo.
echo  Environment setup complete.
echo To activate the virtual environment later, run:
echo     %VENV_DIR%\Scripts\activate.bat
