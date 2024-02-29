#!/bin/bash
sbatch --time=124:00:00 --cpus-per-task=4 --mem-per-cpu=2G -o out/brussoutput.out -e out/bruss.err randbestModel_bruss.sh 
sbatch --time=124:00:00 --cpus-per-task=4 --mem-per-cpu=2G -o out/ellipticoutput.out -e out/elliptic.err randbestModel_elliptic.sh 
sbatch --time=124:00:00 --cpus-per-task=4 --mem-per-cpu=2G -o out/f1output.out -e out/f1.err randbestModel_f1.sh 
sbatch --time=124:00:00 --cpus-per-task=4 --mem-per-cpu=2G -o out/f2output.out -e out/f2.err randbestModel_f2.sh 