#!/bin/bash
#SBATCH --job-name=test_ray_cluster   

# Ray finds and manages all resources on each nodes
# Give access to all resources in one node

#SBATCH --nodes=4                # total number of hyperparam combinations  
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1       

# REMEBER TO SET CPU/GPU VARIABLES BELOW

#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gpus-per-task=1        # number of gpus per task
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node

# Max is 24h for short queue time
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=all
#SBATCH --mail-user=ad2002@princeton.edu

module purge
module load anaconda3/2024.2 
module load cudatoolkit/12.4
conda activate gnode

# Set variables
cpus_per_task=1
gpus_per_task=1

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

# Starting the Ray head node
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus=$cpus_per_task --num-gpus=$gpus_per_task --block &

# Starting the Ray worker nodes

# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus=$cpus_per_task --num-gpus=$gpus_per_task --block &
    sleep 5
done

python -u examples/task_training_scripts/gnode_CDM.py #"$SLURM_CPUS_PER_TASK"
