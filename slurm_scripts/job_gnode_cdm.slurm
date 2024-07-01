#!/bin/bash
#SBATCH --job-name=gnode_test_della       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all
#SBATCH --mail-user=ad2002@princeton.edu

module purge
module load anaconda3/2024.2 
module load cudatoolkit/12.4
conda activate gnode

python examples/task_training_scripts/gnode_CDM.py
