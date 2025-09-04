#!/bin/bash
#SBATCH --job-name=train_ACE_XXXXXX_YYYYYY_CCCCCC
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH -A gfdlhires
####SBATCH --nodes=4
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:2
####SBATCH --time=4:22:00
#SBATCH --time=2:11:00
#SBATCH --output=/scratch4/GFDL/gfdlscr/Linjiong.Zhou/STDO/%x.o%j

set -e
set -x

nlon="XXXXXX"
nlat="YYYYYY"
case="CCCCCC"
data="DDDDDD"

export WANDB_MODE=offline

output_directory="/scratch4/GFDL/gfdlscr/Linjiong.Zhou/datasets/output_directory/${case}"
mkdir -p "${output_directory}"

if [ ! -f "train_config/train_config_${case}.yaml" ]; then
    # Create config
    sed -e "s|XXXXXX|${nlon}|g" \
        -e "s|YYYYYY|${nlat}|g" \
        -e "s|OOOOOO|${output_directory}|g" \
        train_config/train_config_${data}.yaml > train_config/train_config_${case}.yaml
fi

# Activate environment
cd /scratch4/GFDL/gfdlscr/Linjiong.Zhou/ace
source /scratch4/GFDL/gfdlscr/Linjiong.Zhou/miniconda3/etc/profile.d/conda.sh
conda activate fme

# Use srun to launch torchrun on each node
srun --ntasks=${SLURM_NNODES} --ntasks-per-node=1 --cpus-per-task=192 bash -c '
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
  MASTER_IP=$(getent ahostsv4 $MASTER_ADDR | awk "{ print \$1; exit }")
  MASTER_PORT=29500
  PYTHONPATH=/scratch4/GFDL/gfdlscr/Linjiong.Zhou/ace/fme:$PYTHONPATH \
  torchrun \
    --nnodes=${SLURM_NNODES} \
    --nproc_per_node=2 \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_IP}:${MASTER_PORT} \
    -m fme.ace.train train_config/train_config_'${case}'.yaml
'
