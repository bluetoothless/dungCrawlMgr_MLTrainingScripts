@echo off
echo Activating Conda environment...
set env_name=word2world

REM Find Conda path and activate it
REM CALL conda init --all
call conda init cmd.exe

REM Activate the Conda environment
CALL conda activate %env_name%

REM Confirming environment activation
if errorlevel 1 (
    echo Failed to activate environment  >> output_bat.txt
    exit /b 1
)

REM Run 
for /l %%x in (1, 1, 20) do (
    call python .\main.py
)