#!/bin/bash
#SBATCH --job-name=test-job         # name for your job
#SBATCH --partition=short           # partition to run in
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks-per-node=1         # total number of tasks across all nodes<
#SBATCH --time=00:01:00             # total run time limit (HH:MM:SS)
#SBATCH --output=cluster_%j.%x.out  # where to save the output ( %j = JobID, %x = scriptname)

# Optional flags:
##SBATCH --gres gpu:1"              # Generic resource required (here requires 1 GPU)
##SBATCH --mem=100	                # 100 MB RAM per allocated CPU
##SBATCH --error=slurm.%j.err       # where to save error messages ( %j = JobID)
##SBATCH --mail-type=ALL             # send all email
##SBATCH --mail-user=#####          # email address


echo "Unloading modules `hostname`"

# Purge modules to get a pristine environment:
module purge

# Make conda available:
eval "$(conda shell.bash hook)"

# Activate a conda environment:
conda activate dl23


echo "Running python script `hostname`"

# Run a python script:
python test.py
