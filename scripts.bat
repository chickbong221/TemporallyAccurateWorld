@echo off
setlocal

REM ===== Atari100k Experiments =====

echo Running Atari100k Q*bert
set env_name=atari100k-qbert
set run_name=atari100k-qbert
python main.py --wandb

echo Running Atari100k Frostbite
set env_name=atari100k-frostbite
set run_name=atari100k-frostbite
python main.py --wandb

echo Running Atari100k Freeway
set env_name=atari100k-freeway
set run_name=atari100k-freeway
python main.py --wandb


REM ===== DM Control Experiments =====

echo Running DMC Acrobot Swingup
set env_name=dmc-Acrobot-swingup
set run_name=dmc-acrobot-swingup
python main.py --wandb

echo Running DMC Cartpole Swingup Sparse
set env_name=dmc-Cartpole-swingup-sparse
set run_name=dmc-cartpole-swingup-sparse
python main.py --wandb

echo Running DMC Finger Spin
set env_name=dmc-Finger-spin
set run_name=dmc-finger-spin
python main.py --wandb


echo All experiments finished.
pause
