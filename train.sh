#!/bin/bash

#case="r0000" # First run
#case="r0001" # First 2 GPUs run
#case="r0002" # Try 4 GPUs in one node
#case="r0003" # Try 4 GPUs in two nodes
#case="r0004" # same as r0001, add new training data (h500, TMP850, ...)
#case="r0005" # same as r0004, fix extrapolation
#case="r0006" # same as r0005, fix  solar incident
#case="r0007" # same as r0006, use wandb
#case="r0008" # same as r0007, SBATCH --time=17:30:00
#case="r0009" # same as r0008, training data 2023-2024
#case="r0010" # same as r0008, training data 2021-2024
#case="r0011" # same as r0008, training data 2021-2024
#case="r0012" # UFS replay, 1994-2024, more fluxes
#case="r0013" # same as r0012, but for 2 nodes, batch_size:4
#case="r0014" # same as r0012, but for 4 nodes, batch_size:8
#case="r0015" # same as r0014, num_data_workers:32
#case="r0016" # same as r0015, batch_size:16
#case="r0017" # same as r0015, batch_size:24
#case="r0018" # same as r0015, batch_size:32
#case="r0019" # same as r0018, but for 8 nodes
#case="r0020" # new configuration for GFS analysis training
#case="r0021" # new configuration for GFS analysis training
#case="r0022" # new configuration for UFS replay training, num_data_workers:32
#case="r0023" # new configuration for UFS replay training, num_data_workers:32
#case="r0024" # new configuration for UFS replay training, num_data_workers:16
#case="r0025" # new configuration for UFS replay training, num_data_workers:8
#case="r0026" # same as r0021, statistics for all years
#case="r0027" # same as r0024, statistics for all years
#case="r0028" # same as r0024, n_forward_steps:4
#case="r0029" # same as r0024, n_forward_steps:1
#case="r0030" # same as r0027, n_forward_steps:4
#case="r0031" # same as r0027, n_forward_steps:1
#case="r0032" # same as r0027, TMP850:10
#case="r0033" # same as r0027, but for 8 nodes
#case="r0034" # same as r0027, include tendency_of_total_water_path_due_to_advection
#case="r0035" # same as r0034, but for 8 nodes
#case="r0036" # same as r0035, but transfer training from ERA5 (embed_dim: 384)
#case="r0037" # same as r0035, but embed_dim: 384
#case="r0038" # same as r0035, but batch_size: 128 for validation
#case="r0039" # same as r0038, but batch_size: 128 for validation, embed_dim: 384
#case="r0040" # same as r0036, but batch_size: 128 for validation
#case="r0041" # same as r0036, enable parameter_init only in the first epoch
#case="r0042" # same as r0035, but fix the additional data
#case="r0043" # same as r0037, but fix the additional data
#case="r0044" # same as r0036, but fix the additional data
#case="r0045" # same as r0042, but moisture_budget_correction off
#case="r0046" # same as r0043, but moisture_budget_correction off
#case="r0047" # same as r0044, but moisture_budget_correction off
#case="r0048" # same as r0047, but use new file naming
#case="r0049" # same as r0048, but not transfer learning or fine tuning
#case="r0050" # same as r0049, but 25-km training data
#case="r0051" # same as r0050, optimize memory usage
#case="r0052" # same as r0050, optimize memory usage
#case="r0053" # same as r0047, code version: 20250908
#case="r0054" # same as r0048, code version: 20250908
#case="r0055" # same as r0049, code version: 20250908
#case="r0056" # same as r0055, optimize memory usage
#case="r0057" # same as r0055, optimize memory usage
#case="r0058" # same as r0055, optimize memory usage
#case="r0059" # same as r0055, optimize memory usage
#case="r0060" # same as r0055, optimize memory usage
#case="r0061" # same as r0053, embed_dim: 384
#case="r0062" # same as r0053, embed_dim: 512
#case="r0063" # same as r0053, embed_dim: 768
#case="r0064" # same as r0053, scale_factor: 2
#case="r0065" # same as r0053, n_forward_steps: 3
#case="r0066" # same as r0053, scale_factor: 3
#case="r0067" # same as r0053, n_forward_steps: 4
#case="r0068" # same as r0053, change training/validation time period
#case="r0069" # same as r0068, residual_prediction: true
#case="r0070" # same as r0068, fine tuning from ERA5
#case="r0071" # same as r0069, fine tuning from ERA5
#case="r0072" # same as r0070, n_forward_steps: 4
#case="r0073" # same as r0070, n_forward_steps: 8
#case="r0074" # same as r0070, use_gradient_accumulation: true
#case="r0075" # same as r0072, lr: 0.00001, use_gradient_accumulation: true
#case="r0076" # same as r0074, n_forward_steps: 12
#case="r0077" # same as r0072, lr: 0.000001, use_gradient_accumulation: true
#case="r0078" # same as r0075, fine-tune from r0070
#case="r0079" # same as r0072, use_gradient_accumulation: true, fine-tune from r0070
#case="r0080" # same as r0078, n_forward_steps: 8, fine-tune from r0078
#case="r0081" # same as r0079, n_forward_steps: 8, fine-tune from r0079
#case="r0082" # same as r0080, n_forward_steps: 12, fine-tune from r0080
#case="r0083" # same as r0081, n_forward_steps: 12, fine-tune from r0081
#case="r0084" # same as r0077, fine-tune from r0070
#case="r0085" # same as r0075, n_forward_steps: 8, fine-tune from r0075
#case="r0086" # same as r0072, n_forward_steps: 8, fine-tune from r0072
#case="r0087" # same as r0085, n_forward_steps: 12, fine-tune from r0085
#case="r0088" # same as r0086, n_forward_steps: 12, fine-tune from r0086
#case="r0089" # same as r0084, n_forward_steps: 8, fine-tune from r0084
#case="r0090" # same as r0073, lr: 0.00001, use_gradient_accumulation: true
#case="r0091" # same as r0073, lr: 0.000001, use_gradient_accumulation: true
#case="r0092" # same as r0089, n_forward_steps: 12, fine-tune from r0089
#case="r0093" # same as r0076, lr: 0.00001, use_gradient_accumulation: true
#case="r0094" # same as r0076, lr: 0.000001, use_gradient_accumulation: true
#case="r0095" # same as r0077, n_forward_steps: 8, fine-tune from r0077
#case="r0096" # same as r0095, n_forward_steps: 12, fine-tune from r0095
#case="r0097" # same as r0054, for 25-km, 16-layer training, use_gradient_accumulation: true, scale_factor: 3
#case="r0098" # same as r0097, n_forward_steps: 4
#case="r0099" # same as r0097, embed_dim: 384, scale_factor: 4
#case="r0100" # same as r0097, n_forward_steps: 8
#case="r0101" # same as r0098, lr: 0.00001
#case="r0102" # same as r0098, lr: 0.000001
#case="r0103" # same as r0100, lr: 0.00001
#case="r0104" # same as r0100, lr: 0.000001
#case="r0105" # same as r0097, fine-tune from r0097, n_forward_steps: 4, lr: 0.00001
#case="r0106" # same as r0097, fine-tune from r0097, n_forward_steps: 4, lr: 0.000001
#case="r0107" # same as r0105, fine-tune from r0105, n_forward_steps: 8, lr: 0.00001
#case="r0108" # same as r0106, fine-tune from r0106, n_forward_steps: 8, lr: 0.000001
#case="r0109" # same as r0107, fine-tune from r0107, n_forward_steps: 12, lr: 0.00001
#case="r0110" # same as r0108, fine-tune from r0108, n_forward_steps: 12, lr: 0.000001
#case="r0111" # same as r0109, run for 1 epoch, only
#case="r0112" # same as r0110, run for 1 epoch, only
#case="r0113" # same as r0097, scale_factor: 4
#case="r0114" # same as r0097, scale_factor: 4, lr: 0.00001
#case="r0115" # same as r0097, scale_factor: 4, lr: 0.000001
#case="r0116" # same as r0097, fine-tune from r0097, n_forward_steps: 4
#case="r0117" # same as r0116, fine-tune from r0116, n_forward_steps: 8
#case="r0118" # same as r0117, fine-tune from r0117, n_forward_steps: 12
#case="r0119" # same as r0097, fine-tune from r0097, n_forward_steps: 4, lr: 0.0001
#case="r0120" # same as r0097, fine-tune from r0097, n_forward_steps: 4, lr: 0.00001
#case="r0121" # same as r0097, fine-tune from r0097, n_forward_steps: 4, lr: 0.000001
#case="r0122" # same as r0113, fine-tune from r0113, n_forward_steps: 4, lr: 0.0001
#case="r0123" # same as r0113, fine-tune from r0113, n_forward_steps: 4, lr: 0.00001
#case="r0124" # same as r0113, fine-tune from r0113, n_forward_steps: 4, lr: 0.000001
#case="r0125" # same as r0119, fine-tune from r0119, n_forward_steps: 8, lr: 0.0001
#case="r0126" # same as r0120, fine-tune from r0120, n_forward_steps: 8, lr: 0.00001
#case="r0127" # same as r0121, fine-tune from r0121, n_forward_steps: 8, lr: 0.000001
#case="r0128" # same as r0122, fine-tune from r0122, n_forward_steps: 8, lr: 0.0001
#case="r0129" # same as r0123, fine-tune from r0123, n_forward_steps: 8, lr: 0.00001
#case="r0130" # same as r0124, fine-tune from r0124, n_forward_steps: 8, lr: 0.000001
#case="r0131" # same as r0125, fine-tune from r0125, n_forward_steps: 12, lr: 0.0001
#case="r0132" # same as r0126, fine-tune from r0126, n_forward_steps: 12, lr: 0.00001
#case="r0133" # same as r0127, fine-tune from r0127, n_forward_steps: 12, lr: 0.000001
#case="r0134" # same as r0128, fine-tune from r0128, n_forward_steps: 12, lr: 0.0001
#case="r0135" # same as r0129, fine-tune from r0129, n_forward_steps: 12, lr: 0.00001
#case="r0136" # same as r0130, fine-tune from r0130, n_forward_steps: 12, lr: 0.000001
#case="r0137" # same as r0061, fine-tune from r0061, n_forward_steps: 4, lr: 0.0001
#case="r0138" # same as r0061, fine-tune from r0061, n_forward_steps: 4, lr: 0.00001
#case="r0139" # same as r0061, fine-tune from r0061, n_forward_steps: 4, lr: 0.000001
#case="r0140" # same as r0137, fine-tune from r0137, n_forward_steps: 8, lr: 0.0001
#case="r0141" # same as r0138, fine-tune from r0138, n_forward_steps: 8, lr: 0.00001
#case="r0142" # same as r0139, fine-tune from r0139, n_forward_steps: 8, lr: 0.000001
#case="r0143" # same as r0140, fine-tune from r0140, n_forward_steps: 12, lr: 0.0001
#case="r0144" # same as r0141, fine-tune from r0141, n_forward_steps: 12, lr: 0.00001
case="r0145" # same as r0142, fine-tune from r0142, n_forward_steps: 12, lr: 0.000001

#data="gfs"
data="ufs"

nlon=360
nlat=180
#nlon=1440
#nlat=720

# If the first argument is "clean", remove generated files and exit
if [[ "$1" == "clean" ]]; then
    echo "Cleaning up for case ${case}..."
    rm -rf "train_scripts/run_${case}.sh" \
           "train_config/train_config_${case}.yaml" \
           "/scratch4/BIL-P3/bil-coastal-gfdl/Linjiong.Zhou/datasets/output_directory/${case}"
fi

if [ ! -f "train_scripts/run_${case}.sh" ]; then
    sed -e "s|XXXXXX|${nlon}|g" \
        -e "s|YYYYYY|${nlat}|g" \
        -e "s|CCCCCC|${case}|g" \
        -e "s|DDDDDD|${data}|g" \
        train_scripts/run.sh > train_scripts/run_${case}.sh
fi

max_submissions=50
submit_count=0

EXCLUDE_NODES="u22g08"

while [ $submit_count -lt $max_submissions ]; do
    # Is a job with this exact name already in the queue (any state)?
    if ! squeue -u "$USER" -h -o "%j" | grep -qx "train_ACE_${nlon}_${nlat}_${case}"; then
        echo "[$(date)] Job train_ACE_${nlon}_${nlat}_${case} is not in queue. Submitting job #$((submit_count + 1))..."
        if [ -n "$EXCLUDE_NODES" ]; then
            sbatch --exclude=${EXCLUDE_NODES} "train_scripts/run_${case}.sh"
        else
            sbatch "train_scripts/run_${case}.sh"
        fi
        ((submit_count++))
    else
        echo "[$(date)] Job train_ACE_${nlon}_${nlat}_${case} is still in the queue. Waiting..."
    fi
    sleep 600
done

echo "[$(date)] Reached the maximum number of submissions: $max_submissions. Exiting."
