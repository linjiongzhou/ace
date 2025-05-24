#!/bin/bash
#SBATCH --job-name=run_ACE_2024010100_r0002
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/cimes/linjiong/STDO/%x.o%j

set -e
set -x

date="2024010100"
case="r0002"
yyyy="${date:0:4}"
mm="${date:4:2}"
dd="${date:6:2}"
hh="${date:8:2}"

output_directory="/home/linjiong/scratch/ACE2-ERA5_data/output_directory/${date}_${case}"
mkdir -p "${output_directory}"

# Create date-specific config
sed -e "s|OOOOOO|${output_directory}|g" \
    -e "s|YYYY|${yyyy}|g" \
    -e "s|MM|${mm}|g" \
    -e "s|DD|${dd}|g" \
    -e "s|HH|${hh}|g" \
    train_config/train_config.yaml > train_config/train_config_${date}_${case}.yaml

# Activate environment
cd /scratch/cimes/linjiong/ace
source /home/linjiong/miniconda3/etc/profile.d/conda.sh
conda activate fme

# Run torch with 2 processes (1 per GPU)
PYTHONPATH=/scratch/cimes/linjiong/ace/fme:$PYTHONPATH \
torchrun --nproc_per_node=4 \
         --nnodes=1 \
         -m fme.ace.train train_config/train_config_${date}_${case}.yaml

