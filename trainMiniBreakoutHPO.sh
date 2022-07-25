python train.py --algo nhpo --classifier $2 --seed 123 --env mini-breakout-v4 --tensorboard-log $PWD/tensorboard-$2 --hyperparams device:$1
python train.py --algo nhpo --classifier $2 --seed 666 --env mini-breakout-v4 --tensorboard-log $PWD/tensorboard-$2 --hyperparams device:$1
python train.py --algo nhpo --classifier $2 --seed 987 --env mini-breakout-v4 --tensorboard-log $PWD/tensorboard-$2 --hyperparams device:$1
#python train.py --algo hpo --classifier $1 --seed $2 --env mini-breakout-v4 --tensorboard-log $PWD/tensorboards --hyperparams device:$3
#python train.py --algo $1 --classifier $2 --aece WAE --seed $3 --env mini-breakout-v4 --tensorboard-log $PWD/tensorboard --hyperparams device:1
