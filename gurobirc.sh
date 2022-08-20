python train.py --algo hpo --classifier AM --aece WAE --seed 285 --env mini-space_invaders-v4 --tensorboard-log ./tensorboard/AMWAE285gurobiRCRgamma010 --hyperparams device:1
python train.py --algo ppo_time --seed 285 --env mini-space_invaders-v4 --tensorboard-log ./tensorboard/ppot285rewardaddnoisestd050 --hyperparams device:0
python train.py --algo ppo_time --seed 123 --env mini-space_invaders-v4 --tensorboard-log ./tensorboard/ppot123rewardaddnoisestd050 --hyperparams device:0
python train.py --algo ppo_time --seed 987 --env mini-space_invaders-v4 --tensorboard-log ./tensorboard/ppot987rewardaddnoisestd050 --hyperparams device:0
python train.py --algo ppo_time --seed 666 --env mini-space_invaders-v4 --tensorboard-log ./tensorboard/ppot666rewardaddnoisestd050 --hyperparams device:0
python train.py --algo ppo_time --seed 517 --env mini-space_invaders-v4 --tensorboard-log ./tensorboard/ppot517rewardaddnoisestd050 --hyperparams device:0