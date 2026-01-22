#!/usr/bin/env bash
set -e

env_name=atari100k-alien          run_name=Sweep-atari100k python3 main.py --wandb
env_name=atari100k-amidar         run_name=Sweep-atari100k python3 main.py --wandb
env_name=atari100k-assault        run_name=Sweep-atari100k python3 main.py --wandb
env_name=atari100k-asterix        run_name=Sweep-atari100k python3 main.py --wandb
env_name=atari100k-bank_heist     run_name=Sweep-atari100k python3 main.py --wandb
env_name=atari100k-battle_zone    run_name=Sweep-atari100k python3 main.py --wandb
env_name=atari100k-boxing         run_name=Sweep-atari100k python3 main.py --wandb
env_name=atari100k-breakout       run_name=Sweep-atari100k python3 main.py --wandb