#!/bin/bash
#SBATCH --job-name=node_cdm_1_300epoch_wsim       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=4      # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=23:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all
#SBATCH --mail-user=ad2002@princeton.edu

module purge
module load anaconda3/2024.2
conda activate gnode

python examples/task_training_scripts/run_node_cdm.py
