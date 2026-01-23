#!/usr/bin/env bash
set -e

# Various hidden size
env_name=atari100k-qbert run_name=SL-atari100k override_config='{"model_size": "SL"}' python3 main.py --wandb
env_name=atari100k-qbert run_name=SM-atari100k override_config='{"model_size": "SM"}' python3 main.py --wandb
env_name=atari100k-qbert run_name=SS-atari100k override_config='{"model_size": "SS"}' python3 main.py --wandb

# Various horizon size
env_name=atari100k-qbert run_name=H20-atari100k override_config='{"H": 20}' python3 main.py --wandb
env_name=atari100k-qbert run_name=H25-atari100k override_config='{"H": 25}' python3 main.py --wandb
env_name=atari100k-qbert run_name=H30-atari100k override_config='{"H": 30}' python3 main.py --wandb

# Various context size
env_name=atari100k-qbert run_name=context12-atari100k override_config='{"att_context_left": 12}' python3 main.py --wandb
env_name=atari100k-qbert run_name=context16-atari100k override_config='{"att_context_left": 16}' python3 main.py --wandb
env_name=atari100k-qbert run_name=context20-atari100k override_config='{"att_context_left": 20}' python3 main.py --wandb

# loss_sweep
env_name=atari100k-qbert run_name=adv2-atari100k override_config='{"loss_adversarial_scale": 0.2}' python3 main.py --wandb
env_name=atari100k-qbert run_name=adv25-atari100k override_config='{"loss_adversarial_scale": 0.25}' python3 main.py --wandb
env_name=atari100k-qbert run_name=adv30-atari100k override_config='{"loss_adversarial_scale": 0.3}' python3 main.py --wandb
env_name=atari100k-qbert run_name=adv35-atari100k override_config='{"loss_adversarial_scale": 0.35}' python3 main.py --wandb
env_name=atari100k-qbert run_name=contrast0.05-atari100k override_config='{"loss_action_contrast_scale": 0.05}' python3 main.py --wandb
env_name=atari100k-qbert run_name=contrast0.1-atari100k override_config='{"loss_action_contrast_scale": 0.1}' python3 main.py --wandb
env_name=atari100k-qbert run_name=contrast0.15-atari100k override_config='{"loss_action_contrast_scale": 0.15}' python3 main.py --wandb
env_name=atari100k-qbert run_name=contrast0.2-atari100k override_config='{"loss_action_contrast_scale": 0.2}' python3 main.py --wandb