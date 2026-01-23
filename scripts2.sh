set -e

# Various context size
env_name=atari100k-qbert run_name=context12-atari100k override_config='{"att_context_left": 12}' python3 main.py --wandb
env_name=atari100k-qbert run_name=context16-atari100k override_config='{"att_context_left": 16}' python3 main.py --wandb
env_name=atari100k-qbert run_name=context20-atari100k override_config='{"att_context_left": 20}' python3 main.py --wandb

# loss_sweep
env_name=atari100k-qbert run_name=adv2-atari100k override_config='{"loss_adversarial_scale": 0.2}' python3 main.py --wandb
env_name=atari100k-qbert run_name=adv25-atari100k override_config='{"loss_adversarial_scale": 0.25}' python3 main.py --wandb