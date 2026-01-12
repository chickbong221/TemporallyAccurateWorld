@echo off
setlocal enabledelayedexpansion

REM =====================
REM Atari
REM =====================
pip install gym==0.19.0
pip install atari-py==0.2.9
pip install opencv-python==4.7.0.72

REM =====================
REM ROMs
REM =====================
mkdir roms
cd roms

REM Download ROMs using PowerShell
powershell -Command "Invoke-WebRequest -Uri 'http://www.atarimania.com/roms/Roms.rar' -OutFile 'Roms.rar'"

REM Extract ROMs (requires unrar in PATH)
unrar x -o+ Roms.rar

REM Import ROMs
python -m atari_py.import_roms ROMS

REM Cleanup
cd ..
rmdir /s /q roms

REM =====================
REM DMC
REM =====================
pip install dm_control

REM =====================
REM PyTorch
REM =====================
pip install torch==2.0.1
pip install torchvision==0.15.2

REM =====================
REM Numpy
REM =====================
pip install numpy==1.23.5

REM =====================
REM Other
REM =====================
pip install tqdm
pip install wandb
pip install av
pip install tensorboard

echo.
echo Setup complete.
pause
