#!/bin/bash
python train.py --algo ppo --env BreakoutNoFrameskip-v4 --tensorboard-log $PWD/tensorboard
python train.py --algo ppo --env BreakoutNoFrameskip-v4 --tensorboard-log $PWD/tensorboard
