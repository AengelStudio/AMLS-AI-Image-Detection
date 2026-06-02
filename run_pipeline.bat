@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "IMAGE=amls-ai-image-detection:latest"
set "ROOT=%CD%"

echo Building Docker image...
docker build -t "%IMAGE%" "%ROOT%\solution"
if errorlevel 1 exit /b 1

call :run_step python clean.py --timeout_seconds 600
if errorlevel 1 exit /b 1
call :run_step python prepare.py --timeout_seconds 600
if errorlevel 1 exit /b 1
call :run_step python train.py --timeout_seconds 1800
if errorlevel 1 exit /b 1
call :run_step python predict.py --timeout_seconds 600
if errorlevel 1 exit /b 1
call :run_step python train_augmented.py --timeout_seconds 1800
if errorlevel 1 exit /b 1
call :run_step python predict_augmented.py --timeout_seconds 600
if errorlevel 1 exit /b 1

echo.
echo Done. Predictions:
echo   %ROOT%\solution\artifacts\task02\predictions.csv
echo   %ROOT%\solution\artifacts\task03\predictions.csv
exit /b 0

:run_step
echo ^>^>^> %*
docker run --rm --cpus=8 ^
  -v "%ROOT%\data\data:/workspace/solution/data:ro" ^
  -v "%ROOT%\solution\artifacts:/workspace/solution/artifacts" ^
  -w /workspace/solution ^
  "%IMAGE%" %*
exit /b %errorlevel%
