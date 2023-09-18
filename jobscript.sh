#!/bin/bash
#SBATCH --job-name=DeepConvLSTM_01         # name for your job
#SBATCH --partition=gpu           # partition to run in
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks-per-node=64         # total number of tasks across all nodes<
#SBATCH --time=24:00:00             # total run time limit (HH:MM:SS)
#SBATCH --output=cluster_%j.%x.out  # where to save the output ( %j = JobID, %x = scriptname)
#SBATCH --gres gpu:1              # Generic resource required (1/2/4)
#SBATCH --mail-type=ALL             # send all email
#SBATCH --mail-user=$EMAIL          # email address from environemnt variable $EMAIL

#Optional parameters
##SBATCH --mem=100	                # 100 MB RAM per allocated CPU
##SBATCH --error=slurm.%j.err       # where to save error messages ( %j = JobID)



echo "Unloading modules `hostname`"

# Purge modules to get a pristine environment:
module purge

# Make conda available:
eval "$(conda shell.bash hook)"

# Activate a conda environment:
conda activate dl23


echo "Running python script `hostname`"

# Run a python script:
python main.py configs/default.json
