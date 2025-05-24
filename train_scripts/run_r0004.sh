#!/bin/bash
#SBATCH --job-name=train_ACE_r0004
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/cimes/linjiong/STDO/%x.o%j

set -e
set -x

case="r0004"

output_directory="/home/linjiong/scratch/ACE2-ERA5_data/output_directory/${case}"
mkdir -p "${output_directory}"

# Create config
sed -e "s|OOOOOO|${output_directory}|g" \
    train_config/train_config.yaml > train_config/train_config_${case}.yaml

# Activate environment
cd /scratch/cimes/linjiong/ace
source /home/linjiong/miniconda3/etc/profile.d/conda.sh
conda activate fme

# Run torch with 2 processes (1 per GPU)
PYTHONPATH=/scratch/cimes/linjiong/ace/fme:$PYTHONPATH \
torchrun --nproc_per_node=2 \
         --nnodes=1 \
         -m fme.ace.train train_config/train_config_${case}.yaml

