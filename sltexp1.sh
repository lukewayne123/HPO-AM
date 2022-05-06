# python train.py --algo hpo --classifier AM --aece WAE --seed 666 --env mini-breakout-v4 --tensorboard-log /home/lcouy/HDD/HPO-AM/tensorboard/AMWAEslt666rgamma005 --hyperparams rgamma:0.05
# python train.py --algo hpo --classifier AM --aece WAE --seed 123 --env mini-breakout-v4 --tensorboard-log /home/lcouy/HDD/HPO-AM/tensorboard/AMWAEslt123rgamma010 --hyperparams rgamma:0.10
python train.py --algo hpo --classifier AM --aece WAE --seed 456 --env mini-breakout-v4 --tensorboard-log /home/lcouy/HDD/HPO-AM/tensorboard/AMWAEslt456rgamma010 --hyperparams rgamma:0.10 device:0

