#!/bin/bash
sbatch --time=196:00:00 -o out/brussoutput.out -e out/bruss.err bestModel_bruss.sh 
sbatch --time=196:00:00 -o out/ellipticoutput.out -e out/elliptic.err bestModel_elliptic.sh 
sbatch --time=196:00:00 -o out/f1output.out -e out/f1.err bestModel_f1.sh 
sbatch --time=196:00:00 -o out/f2output.out -e out/f2.err bestModel_f2.sh 