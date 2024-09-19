@echo off

REM Check if running as administrator
NET SESSION >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO Please run this script as administrator.
    PAUSE
    EXIT /B 1
)

REM Install Chocolatey if not installed
where choco >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO Installing Chocolatey...
    @powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-ExecutionPolicy Bypass -Scope Process; [System.Net.ServicePointManager]::SecurityProtocol = 'Tls12'; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"
)

REM Update Chocolatey
choco upgrade chocolatey -y

REM Install Git if not installed
where git >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    ECHO Installing Git...
    choco install git -y
)

REM Install Python 3.12
ECHO Installing Python 3.12...
choco install python --version=3.12 -y

REM Update PATH to include Python 3.12
SET "PATH=C:\Python312\Scripts\;C:\Python312\;%PATH%"

REM Verify Python 3.12 is installed
python --version 2>nul | find "3.12" >nul
IF %ERRORLEVEL% NEQ 0 (
    ECHO Python 3.12 not found in PATH.
    EXIT /B 1
)

REM Install CMake
ECHO Installing CMake...
choco install cmake -y

REM Install Visual Studio Build Tools (includes MSVC compiler)
ECHO Installing Visual Studio Build Tools...
choco install visualstudio2022buildtools --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools" --accept-license -y

REM Install GoogleTest
ECHO Installing GoogleTest...
IF NOT EXIST googletest (
    git clone https://github.com/google/googletest.git
)

REM Build GoogleTest
cd googletest
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=C:\gtest
cmake --build . --config Release --target INSTALL
cd ..\..

REM Set GTEST_ROOT environment variable
SETX GTEST_ROOT "C:\gtest"

REM Go back to the original directory
cd %~dp0

REM Create and activate a Python 3.12 virtual environment
ECHO Creating Python 3.12 virtual environment...

python -m venv venv

REM Activate the virtual environment
CALL venv\Scripts\activate.bat

ECHO.
ECHO Python 3.12 environment is ready and activated.
ECHO.

PAUSE