#!/bin/bash
sbatch --time=27-0:0:0 --cpus-per-task=2 --mem-per-cpu=1G -o outInc/brussoutput.out -e out/bruss.err inc_bruss.sh
sbatch --time=27-0:0:0 --cpus-per-task=2 --mem-per-cpu=1G -o outInc/ellipticoutput.out -e out/elliptic.err inc_elliptic.sh
sbatch --time=27-0:0:0 --cpus-per-task=2 --mem-per-cpu=1G -o outInc/f1output.out -e out/f1.err inc_f1.sh
sbatch --time=27-0:0:0 --cpus-per-task=2 --mem-per-cpu=1G -o outInc/f2output.out -e out/f2.err inc_f2.sh 