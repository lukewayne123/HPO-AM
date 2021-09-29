python train.py --algo ppo  --seed 517 --env mini-breakout-v4 --tensorboard-log $PWD/tensorboard --hyperparams device:1
python train.py --algo a2c  --seed 123 --env mini-breakout-v4 --tensorboard-log $PWD/tensorboard --hyperparams device:1
python train.py --algo a2c  --seed 666 --env mini-breakout-v4 --tensorboard-log $PWD/tensorboard --hyperparams device:0