#!/bin/bash
module load python/3.9
source ./ENV/bin/activate
pip install --no-index --upgrade pip
pip install -r requirements.txt
python python comparison.py 