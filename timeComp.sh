#!/bin/bash
module load python/3.9
virtualenv --no-download ENV
source ENV/bin/activate
pip install --no-index --upgrade pip
pip install -r requirements.txt
python timeComparisonPOF.py