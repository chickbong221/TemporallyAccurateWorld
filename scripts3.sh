#!/usr/bin/env bash
set -e

env_name=atari100k-qbert run_name=adv30-atari100k override_config='{"loss_adversarial_scale": 0.3}' python3 main.py --wandb
env_name=atari100k-qbert run_name=adv35-atari100k override_config='{"loss_adversarial_scale": 0.35}' python3 main.py --wandb
env_name=atari100k-qbert run_name=contrast0.05-atari100k override_config='{"loss_action_contrast_scale": 0.05}' python3 main.py --wandb
env_name=atari100k-qbert run_name=contrast0.1-atari100k override_config='{"loss_action_contrast_scale": 0.1}' python3 main.py --wandb
env_name=atari100k-qbert run_name=contrast0.15-atari100k override_config='{"loss_action_contrast_scale": 0.15}' python3 main.py --wandb
env_name=atari100k-qbert run_name=contrast0.2-atari100k override_config='{"loss_action_contrast_scale": 0.2}' python3 main.py --wandb