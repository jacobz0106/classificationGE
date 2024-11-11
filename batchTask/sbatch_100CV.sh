#!/bin/bash
sbatch --time=20-23:00:00 --cpus-per-task=2 --mem-per-cpu=1G -o out/brussoutput.out -e out/bruss.err bestModel_bruss.sh 
sbatch --time=27-23:00:00 --cpus-per-task=2 --mem-per-cpu=1G -o out/ellipticoutput.out -e out/elliptic.err bestModel_elliptic.sh 
sbatch --time=20-23:00:00 --cpus-per-task=2 --mem-per-cpu=1G -o out/f1output.out -e out/f1.err bestModel_f1.sh 
sbatch --time=20-23:00:00 --cpus-per-task=2 --mem-per-cpu=1G -o out/f2output.out -e out/f2.err bestModel_f2.sh 