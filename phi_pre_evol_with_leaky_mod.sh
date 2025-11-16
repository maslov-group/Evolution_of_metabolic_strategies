#!/bin/bash
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --job-name=phi_pre_2res
#SBATCH --error=logs/cout_param_sweep_%A_%a.err
#SBATCH --output=logs/cout_param_sweep_%A_%a.out
#SBATCH --array=0-219
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G

module purge
module load Python/3.10.1-IGB-gcc-8.2.0
python phi_pre_evol_with_leaky_mod.py $SLURM_ARRAY_TASK_ID