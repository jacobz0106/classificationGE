#!/bin/bash
module load python/3.9
module load gurobi
source ~/env_gurobi/bin/activate
pip install --no-index --upgrade pip
pip install -r requirements.txt
python comparison.py