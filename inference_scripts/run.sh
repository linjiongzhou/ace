#!/bin/bash
#SBATCH --job-name=run_ACE_XXXXXX_YYYYYY_CCCCCC_TTTTTT_PPPPPP_DDDDDD
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH -A gfdlhires
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=0:30:00
#SBATCH --cpus-per-task=6
#SBATCH --output=/scratch4/GFDL/gfdlhires/Linjiong.Zhou/STDO/%x.o%j

set -euo pipefail
set -x

nlon="XXXXXX"
nlat="YYYYYY"
case="CCCCCC"
data="TTTTTT"
ckpt="PPPPPP"
date="DDDDDD"
yyyy="${date:0:4}"
mm="${date:4:2}"
dd="${date:6:2}"
hh="${date:8:2}"

if [[ "${ckpt}" == "ERA5" ]]; then
    ckpt_file="/scratch4/GFDL/gfdlscr/Linjiong.Zhou/datasets/ace2_era5_ckpt.tar"
fi
if [[ "${ckpt}" == "UFS" || "${ckpt}" == "GFS" ]]; then
    ckpt_file="/scratch4/BIL-P3/bil-coastal-gfdl/Linjiong.Zhou/datasets/output_directory/${case}/training_checkpoints/best_ckpt.tar"
    # ckpt_file="/scratch4/BIL-P3/bil-coastal-gfdl/Linjiong.Zhou/datasets/output_directory/${case}/training_checkpoints/best_inference_ckpt.tar"
fi

output_directory="/scratch4/BIL-P3/bil-coastal-gfdl/Linjiong.Zhou/datasets/output_directory/${case}_${data}_${ckpt}_${date}"
mkdir -p "${output_directory}"

# ---- Match training node software ----
module purge
module load cuda/12.9.1

# ---- Set timestamp depending on grid size ----
if [[ "${nlon}x${nlat}" == "360x180" ]]; then
    timestamp="${yyyy}"
elif [[ "${nlon}x${nlat}" == "1440x720" ]]; then
    timestamp="${yyyy}${mm}"
fi

# ---- Make run-specific YAML ----
sed -e "s|XXxYY|${nlon}x${nlat}|g" \
    -e "s|FFFFFF|${ckpt_file}|g" \
    -e "s|OOOOOO|${output_directory}|g" \
    -e "s|TTTSSS|${timestamp}|g" \
    -e "s|YYYY|${yyyy}|g" \
    -e "s|MM|${mm}|g" \
    -e "s|DD|${dd}|g" \
    -e "s|HH|${hh}|g" \
    inference_config/inference_config_${data}.yaml > inference_config/inference_config_${case}_${data}_${ckpt}_${date}.yaml

# ---- Activate environment ----
cd /scratch4/GFDL/gfdlhires/Linjiong.Zhou/ace
source /scratch4/GFDL/gfdlscr/Linjiong.Zhou/miniconda3/etc/profile.d/conda.sh
set +u
conda activate fme
set -u

# ---- Keep RAM usage sane & avoid compilation memory spikes ----
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ensure project is importable even if PYTHONPATH was empty
export PYTHONPATH="/scratch4/GFDL/gfdlhires/Linjiong.Zhou/ace/fme${PYTHONPATH:+:$PYTHONPATH}"

# ---- Run inference (single process) ----
python -m fme.ace.inference inference_config/inference_config_${case}_${data}_${ckpt}_${date}.yaml

# ---- Rename output for convenience ----
if [ -f "${output_directory}/autoregressive_predictions.nc" ]; then
    cp -f "${output_directory}/autoregressive_predictions.nc" \
          "${output_directory}/predictions_${case}_${data}_${ckpt}_${date}.nc"
else
    echo "Output file not found."
    exit 1
fi
