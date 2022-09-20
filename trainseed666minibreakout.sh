python train.py --algo a2c  --seed 666 --env mini-breakout-v4 --tensorboard-log $PWD/tensorboard --hyperparams device:0
python train.py --algo ppo  --seed 666 --env mini-breakout-v4 --tensorboard-log $PWD/tensorboard --hyperparams device:0
python train.py --algo hpo --entropy-hpo --classifier AM --aece WAE --seed 666 --env mini-breakout-v4 --tensorboard-log $PWD/tensorboard --hyperparams device:0
python train.py --algo hpo --entropy-hpo --classifier AM-log --aece WAE --seed 666 --env mini-breakout-v4 --tensorboard-log $PWD/tensorboard --hyperparams device:0
python train.py --algo hpo --entropy-hpo --classifier AM-root --aece WAE --seed 666 --env mini-breakout-v4 --tensorboard-log $PWD/tensorboard --hyperparams device:0
python train.py --algo hpo --entropy-hpo --classifier AM-square --aece WAE --seed 666 --env mini-breakout-v4 --tensorboard-log $PWD/tensorboard --hyperparams device:0
python train.py --algo hpo --entropy-hpo --classifier AM-sub --aece WAE --seed 666 --env mini-breakout-v4 --tensorboard-log $PWD/tensorboard --hyperparams device:0