echo "fix not env set at begining , smaller learning rate tmux 5 fix advantage, advantage use from action_advantages"
CUDA_LAUNCH_BLOCKING=1 python train.py --algo hpo --classifier AM-log --env gridworld_randR_env-v0 --tensorboard-log $PWD/tensorboard --hyperparams device:1
echo "smaller learning rate tmux 5"