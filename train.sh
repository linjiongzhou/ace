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
case="r0051" # same as r0050, optimize memory usage
#case="r0052" # same as r0050, optimize memory usage

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
           "../datasets/output_directory/${case}"
fi

if [ ! -f "train_scripts/run_${case}.sh" ]; then
    sed -e "s|XXXXXX|${nlon}|g" \
        -e "s|YYYYYY|${nlat}|g" \
        -e "s|CCCCCC|${case}|g" \
        -e "s|DDDDDD|${data}|g" \
        train_scripts/run.sh > train_scripts/run_${case}.sh
fi

max_submissions=10
submit_count=0

while [ $submit_count -lt $max_submissions ]; do
    # Is a job with this exact name already in the queue (any state)?
    if ! squeue -u "$USER" -h -o "%j" | grep -qx "train_ACE_${nlon}_${nlat}_${case}"; then
        echo "[$(date)] Job train_ACE_${nlon}_${nlat}_${case} is not in queue. Submitting job #$((submit_count + 1))..."
        sbatch "train_scripts/run_${case}.sh"   # <-- no nodelist / no exclude
        ((submit_count++))
    else
        echo "[$(date)] Job train_ACE_${nlon}_${nlat}_${case} is still in the queue. Waiting..."
    fi
    sleep 600
done

echo "[$(date)] Reached the maximum number of submissions: $max_submissions. Exiting."
