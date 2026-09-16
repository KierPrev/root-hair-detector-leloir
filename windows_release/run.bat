@echo off
setlocal enabledelayedexpansion

if "%~1"=="" (
    echo Arrastra una o mas imagenes .tif sobre este archivo para procesarlas.
    echo.
    pause
    exit /b
)

for %%F in (%*) do (
    echo.
    echo Procesando %%~nxF ...
    powershell -NoProfile -Command "Measure-Command { & '%~dp0hair-detection.exe' '%%F' } | Select-Object TotalSeconds"
)

echo.
echo Listo. Resultados (CSV + overlay) en la carpeta data\results de donde corriste el programa.
pause
