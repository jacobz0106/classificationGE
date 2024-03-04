#!/bin/bash
module load gurobi/10.0.3 python/3.9
source ~/env_gurobi/bin/activate
pip install --no-index --upgrade pip
pip install -r ../requirements.txt
export OMP_NUM_THREADS=1
export GRB_TOKENSERVER=license1.computecanada.ca
echo "installation of req packages done."
python ../best_model_gridsearchCV.py $TRAIN_SIZE 500 Function1 POF
