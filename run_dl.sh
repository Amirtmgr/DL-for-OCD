#!/bin/bash

# Submit DL SLURM jobs

# Task 2: cHW vs rHW
sbatch jobscripts/dl_scripts/sgd/task_02_dl.sh
sbatch jobscripts/dl_scripts/adam/task_02_dl.sh


# With SGD
sbatch jobscripts/dl_scripts/sgd/task_01_dl.sh
sbatch jobscripts/dl_scripts/sgd/task_03_dl.sh
sbatch jobscripts/dl_scripts/sgd/task_04_dl.sh

# With ADAM
sbatch jobscripts/dl_scripts/adam/task_01_dl.sh
sbatch jobscripts/dl_scripts/adam/task_03_dl.sh
sbatch jobscripts/dl_scripts/adam/task_04_dl.sh
