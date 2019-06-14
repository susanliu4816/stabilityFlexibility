#!/bin/bash
#SBATCH --array=1-73
#SBATCH -n 1
#SBATCH -c 1
#SBATCH -t 0:15:00
#SBATCH --mem=1GB
#SBATCH -J stabilityFlexibility
#SBATCH -o optimalGains/slurm-%A-%a.out
#SBATCH --mail-type=all
#SBATCH --mail-user=shuningl@princeton.edu



module load anacondapy
source activate PsyNeuLink

FILENAME="${SLURM_ARRAY_TASK_ID}.csv"


python -u stabilityflexibilitytesting.py $FILENAME

