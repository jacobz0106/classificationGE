#!/bin/bash
module load gurobi/10.0.3 python/3.9
source ~/env_gurobi/bin/activate
pip install --no-index --upgrade pip
pip install -r ../requirements.txt
export GRB_TOKENSERVER=license1.computecanada.ca
echo "installation of req packages done."
python ../testBest_model.py 100 500 Function1 POF
