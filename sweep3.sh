#!/usr/bin/env bash
set -e

env_name=atari100k-krull          run_name=Sweep-atari100k python3 main.py --wandb
env_name=atari100k-kung_fu_master run_name=Sweep-atari100k python3 main.py --wandb
env_name=atari100k-ms_pacman      run_name=Sweep-atari100k python3 main.py --wandb
env_name=atari100k-pong           run_name=Sweep-atari100k python3 main.py --wandb
env_name=atari100k-private_eye    run_name=Sweep-atari100k python3 main.py --wandb
env_name=atari100k-qbert          run_name=Sweep-atari100k python3 main.py --wandb
env_name=atari100k-road_runner    run_name=Sweep-atari100k python3 main.py --wandb
env_name=atari100k-seaquest       run_name=Sweep-atari100k python3 main.py --wandb
env_name=atari100k-up_n_down      run_name=Sweep-atari100k python3 main.py --wandb