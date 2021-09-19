python train.py --algo hpo --classifier $1 --seed $2 --env mini-asterix-v4 --tensorboard-log $PWD/tensorboards --hyperparams device:$3
#python train.py --algo $1 --classifier $2 --aece WAE --seed $3 --env mini-breakout-v4 --tensorboard-log $PWD/tensorboard --hyperparams device:1
