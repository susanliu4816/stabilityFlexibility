#!/bin/bash
#SBATCH --array=29
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 3:30:00
#SBATCH --mem=2GB
#SBATCH -J stabilityFlexibility
#SBATCH -o subject29/slurm-%A-%a.out
#SBATCH --mail-type=all
#SBATCH --mail-user=shuningl@princeton.edu

module load anacondapy
source activate PsyNeuLink

FILENAME="${SLURM_ARRAY_TASK_ID}.csv"


python -u StabilityFlexibility.py $FILENAME

