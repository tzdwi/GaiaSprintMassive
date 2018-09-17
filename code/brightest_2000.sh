#!/bin/sh
#SBATCH --job-name=b2000
#SBATCH --account=stf
#SBATCH --partition=stf
#SBATCH --nodes=1
#SBATCH --workdir=/usr/lusers/tzdw/Research/code/WISE_features/
#SBATCH --mem=120G
#SBATCH --ntasks-per-node=28
#SBATCH --time=100:00:00
#SBATCH --mail-user="trevorzaylen@gmail.com"
#SBATCH --mail-type=ALL

export PATH="/usr/lusers/tzdw/miniconda2/bin:$PATH"

python assemble_features.py /gscratch/stf/tzdw/WISE_lcs/brightest_2000/data/ b2000_features.csv 28 3600
