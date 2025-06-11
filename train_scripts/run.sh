#!/bin/bash
#SBATCH --job-name=train_ACE_CCCCCC
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH -A gfdlhires
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:h100:2
#SBATCH --time=17:30:00
#SBATCH --output=/scratch4/GFDL/gfdlscr/Linjiong.Zhou/STDO/%x.o%j

set -e
set -x

case="CCCCCC"

export WANDB_MODE=offline

output_directory="/scratch4/GFDL/gfdlscr/Linjiong.Zhou/datasets/output_directory/${case}"
mkdir -p "${output_directory}"

# Create config
sed -e "s|OOOOOO|${output_directory}|g" \
    -e "s|NNNNNN|training_${case}|g" \
    train_config/train_config.yaml > train_config/train_config_${case}.yaml

# Activate environment
cd /scratch4/GFDL/gfdlscr/Linjiong.Zhou/ace
source /scratch4/GFDL/gfdlscr/Linjiong.Zhou/miniconda3/etc/profile.d/conda.sh
conda activate fme

# Run torch with 2 processes (1 per GPU)
PYTHONPATH=/scratch4/GFDL/gfdlscr/Linjiong.Zhou/ace/fme:$PYTHONPATH \
torchrun --nproc_per_node=2 \
         --nnodes=1 \
         -m fme.ace.train train_config/train_config_${case}.yaml

#wandb sync $output_directory/wandb/latest-run
