#!/bin/bash
#SBATCH --job-name=run_ACE_CCCCCC_DDDDDD
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/cimes/linjiong/STDO/%x.o%j
#SBATCH --gres=gpu:1

set -e
set -x

case="CCCCCC"
date="DDDDDD"
yyyy="${date:0:4}"
mm="${date:4:2}"
dd="${date:6:2}"
hh="${date:8:2}"

output_directory="/home/linjiong/scratch/datasets/output_directory/${case}_${date}"
ckpt_file="/home/linjiong/scratch/datasets/output_directory/${case}/training_checkpoints/best_inference_ckpt.tar"

mkdir -p "${output_directory}"

sed -e "s|FFFFFF|${ckpt_file}|g" \
    -e "s|OOOOOO|${output_directory}|g" \
    -e "s|YYYY|${yyyy}|g" \
    -e "s|MM|${mm}|g" \
    -e "s|DD|${dd}|g" \
    -e "s|HH|${hh}|g" \
    inference_config/inference_config.yaml > inference_config/inference_config_${case}_${date}.yaml

cd /scratch/cimes/linjiong/ace
source /home/linjiong/miniconda3/etc/profile.d/conda.sh
conda activate fme

PYTHONPATH=/scratch/cimes/linjiong/ace/fme:$PYTHONPATH \
python -m fme.ace.inference inference_config/inference_config_${case}_${date}.yaml

if [ -f "${output_directory}/autoregressive_predictions.nc" ]; then
    cp -rf "${output_directory}/autoregressive_predictions.nc" \
       "${output_directory}/predictions_${case}_${date}.nc"
else
    echo "Output file not found."
    exit 1
fi
