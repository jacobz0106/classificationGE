#!/bin/bash
virtualenv --no-download ENV
source ENV/bin/activate
module load python/3.9
pip install --no-index --upgrade pip
pip install -r requirements.txt
python timeComparisonPOF.py