#!/bin/bash

##################################
pkill -9 Carla
pkill -9 python
##################################


conda activate camo
sleep 1

export ROOT=/home/adarash/camouflage
export CARLA_ROOT=/home/adarash/CARLA

export PORT=2000


export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla":$PYTHONPATH
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg":$PYTHONPATH


# launch carla world (commented out - not needed for renderer training)
# $CARLA_ROOT/CarlaUE4.sh -quality-level=low -world-port=$PORT -RenderOffScreen &

#$CARLA_ROOT/CarlaUE4.sh -quality-level=low -world-port=$PORT -resx=800 -resy=600 &
# PID=$!
# echo "Carla PID=$PID"

# sleep 10

# launch script

# Dataset generation (completed)
# python car_segmentation.py --output-dir dataset_8k/train --num-samples 6400 --resume
# python car_segmentation.py --output-dir dataset_8k/val --num-samples 1600 --resume

# Train neural renderer (batch 10 uses ~19.4GB on RTX 3090)
python models/train_renderer.py --dataset dataset_8k/train --val-dataset dataset_8k/val --epochs 100 --batch-size 10
