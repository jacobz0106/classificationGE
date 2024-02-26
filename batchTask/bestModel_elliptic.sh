#!/bin/bash
module load gurobi/10.0.3 python/3.9
source ~/env_gurobi/bin/activate
pip install --no-index --upgrade pip
pip install -r ../requirements.txt
#export OMP_NUM_THREADS=1
python ../best_model_gridsearchCV_elliptic.py
