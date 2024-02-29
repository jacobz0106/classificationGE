#!/bin/bash
sbatch --time=124:00:00 --cpus-per-task=2 --mem-per-cpu=1G -o out/brussoutputRand.out -e out/brussRand.err randbestModel_bruss.sh 
sbatch --time=124:00:00 --cpus-per-task=2 --mem-per-cpu=1G -o out/ellipticoutputRand.out -e out/ellipticRand.err randbestModel_elliptic.sh 
sbatch --time=124:00:00 --cpus-per-task=2 --mem-per-cpu=1G -o out/f1outputRand.out -e out/f1Rand.err randbestModel_f1.sh 
sbatch --time=124:00:00 --cpus-per-task=2 --mem-per-cpu=1G -o out/f2outputRand.out -e out/f2Rand.err randbestModel_f2.sh 