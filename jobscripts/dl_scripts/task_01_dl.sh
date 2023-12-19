#!/bin/bash
#SBATCH --job-name=DL_10         # name for your job
#SBATCH --partition=gpu           # partition to run in
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --ntasks-per-node=64        # total number of tasks across all nodes<
#SBATCH --time=24:00:00             # total run time limit (HH:MM:SS)
#SBATCH --output=cluster/omni_%x_%j.out  # where to save the output ( %j = JobID, %x = scriptname)
#SBATCH --error=cluster/omni_%x_%j.err       # where to save error messages ( %j = JobID)
#SBATCH --gres gpu:1              # Generic resource required (1/2/4)
#SBATCH --mem=75G	                # 100 MB RAM per allocated CPU

#Optional parameters

##SBATCH --mail-type=ALL             # send all email
##SBATCH --mail-user=$EMAIL          # email address from environemnt variable $EMAIL


# Python args
method="dl" # ml, dl
task="task1" # task1, task2, task3, task4, task5


echo "Unloading modules `hostname`"

# Purge modules to get a pristine environment:
module purge

# Make conda available:
eval "$(conda shell.bash hook)"

# Activate a conda environment:
conda activate ml23


echo "JobID is $SLURM_JOBID"
echo "Working directory is $SLURM_SUBMIT_DIR"
echo "Running python script `hostname`"


# Run a python script:
python main.py configs/$task.yaml $method

