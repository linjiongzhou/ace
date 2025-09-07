#!/bin/bash
#SBATCH --job-name=train_ACE_XXXXXX_YYYYYY_CCCCCC
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH -A gfdlhires
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1          # one Slurm task per node
#SBATCH --gres=gpu:h100:2            # both GPUs to that task
#SBATCH --time=24:00:00
#SBATCH --exclude=u22g08
#SBATCH --output=/scratch4/GFDL/gfdlscr/Linjiong.Zhou/STDO/%x.o%j

set -euo pipefail
set -x

# ---- Fill these per run ----
nlon="XXXXXX"
nlat="YYYYYY"
case="CCCCCC"
data="DDDDDD"

export WANDB_MODE=offline

output_directory="/scratch4/GFDL/gfdlscr/Linjiong.Zhou/datasets/output_directory/${case}"
mkdir -p "${output_directory}"

if [ ! -f "train_config/train_config_${case}.yaml" ]; then
  sed -e "s|XXxYY|${nlon}x${nlat}|g" \
      -e "s|OOOOOO|${output_directory}|g" \
      "train_config/train_config_${data}.yaml" > "train_config/train_config_${case}.yaml"
fi

# ---- Ursa modules ----
module purge
module load cuda/12.9.1

# ---- Repo / env ----
cd /scratch4/GFDL/gfdlscr/Linjiong.Zhou/ace
source /scratch4/GFDL/gfdlscr/Linjiong.Zhou/miniconda3/etc/profile.d/conda.sh
conda activate fme

# ---- NCCL hints (IB) ----
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
# If IB gives trouble, uncomment:
# export NCCL_IB_DISABLE=1

# ---- When training on high-resolution data ----
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF="backend:cudaMallocAsync,max_split_size_mb:256,garbage_collection_threshold:0.8"
export HDF5_USE_FILE_LOCKING=FALSE

# ---- Launch: 1 task per node; torchrun spawns 2 procs (one per GPU) ----
srun \
  --ntasks=${SLURM_NNODES} \
  --ntasks-per-node=1 \
  --gpus-per-task=2 \
  --gpu-bind=map_gpu:0,1 \
  --cpus-per-task=192 \
bash -lc '
  source /scratch4/GFDL/gfdlscr/Linjiong.Zhou/miniconda3/etc/profile.d/conda.sh
  conda activate fme

  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  # Make BOTH GPUs visible to this task (e.g., "0,1"); avoids device=1 num_gpus=1 errors
  export CUDA_VISIBLE_DEVICES=${SLURM_STEP_GPUS}

  echo "Node=$(hostname)  SLURM_STEP_GPUS=${SLURM_STEP_GPUS}  CVD=$CUDA_VISIBLE_DEVICES"
  nvidia-smi -L || true
python - <<PY
import os, torch
print("Sanity:", torch.__version__, "CUDA", torch.version.cuda,
      "| visible_gpus =", torch.cuda.device_count(),
      "| CVD =", os.environ.get("CUDA_VISIBLE_DEVICES"))
PY

  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
  MASTER_IP=$(getent ahostsv4 $MASTER_ADDR | awk "{ print \$1; exit }")
  MASTER_PORT=29500

  PYTHONPATH=/scratch4/GFDL/gfdlscr/Linjiong.Zhou/ace/fme:$PYTHONPATH \
  torchrun \
    --nnodes=${SLURM_NNODES} \
    --nproc_per_node=2 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_IP}:${MASTER_PORT} \
    --rdzv_id=${SLURM_JOB_ID} \
    --rdzv-conf="timeout=600" \
    -m fme.ace.train train_config/train_config_'${case}'.yaml
'
