#!/bin/bash
sbatch --time=29-0:0:0 --cpus-per-task=2 --mem-per-cpu=1G -o out/brussoutputRand.out -e out/brussRand.err inc_brussRand.sh
sbatch --time=29-0:0:0 --cpus-per-task=2 --mem-per-cpu=1G -o out/ellipticoutputRand.out -e out/ellipticRand.err inc_ellipticRand.sh
sbatch --time=29-0:0:0 --cpus-per-task=2 --mem-per-cpu=1G -o out/f1outputRand.out -e out/f1Rand.err inc_f1Rand.sh
sbatch --time=29-0:0:0 --cpus-per-task=2 --mem-per-cpu=1G -o out/f2outputRand.out -e out/f2Rand.err inc_f2Rand.sh