@echo off
REM Batch script to build SymSorter Windows executable
REM This script will install PyInstaller and build the exe

echo ========================================
echo   SymSorter Executable Builder
echo ========================================
echo.

REM Check if conda environment is activated
if "%CONDA_DEFAULT_ENV%"=="" (
    echo ERROR: No conda environment is activated!
    echo Please run: conda activate symsorter
    echo.
    pause
    exit /b 1
)

echo Current environment: %CONDA_DEFAULT_ENV%
echo.

REM Install PyInstaller if not already installed
echo Installing PyInstaller...
pip install pyinstaller --quiet
if errorlevel 1 (
    echo ERROR: Failed to install PyInstaller
    pause
    exit /b 1
)

echo.
echo ========================================
echo Building executable...
echo This may take 5-10 minutes...
echo ========================================
echo.

REM Run the build script
python build_exe.py

if errorlevel 1 (
    echo.
    echo ========================================
    echo BUILD FAILED!
    echo ========================================
    pause
    exit /b 1
)

echo.
echo ========================================
echo BUILD SUCCESSFUL!
echo ========================================
echo.
echo Your executable is ready at:
echo   dist\SymSorter.exe
echo.
echo File size: Large (1-2 GB due to PyTorch)
echo.
echo You can now distribute this .exe file to users!
echo.
pause

