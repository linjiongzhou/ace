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
case="r0077" # same as r0072, lr: 0.000001, use_gradient_accumulation: true
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

data="ufs"
#data="gfs"
#data="ifs"

ckpt="UFS"
#ckpt="GFS"
#ckpt="ERA5"

nlon=360
nlat=180
#nlon=1440
#nlat=720

dates=(\
		# ufs
        "2020010100" "2020011100" "2020012100" \
        "2020020100" "2020021100" "2020022100" \
        "2020030100" "2020031100" "2020032100" \
        "2020040100" "2020041100" "2020042100" \
        "2020050100" "2020051100" "2020052100" \
        "2020060100" "2020061100" "2020062100" \
        "2020070100" "2020071100" "2020072100" \
        "2020080100" "2020081100" "2020082100" \
        "2020090100" "2020091100" "2020092100" \
        "2020100100" "2020101100" "2020102100" \
        "2020110100" "2020111100" "2020112100" \
        "2020120100" "2020121100" "2020122100" \
        # \
		# gfs part 2, ufs
        "2022010100" "2022011100" "2022012100" \
        "2022020100" "2022021100" "2022022100" \
        "2022030100" "2022031100" "2022032100" \
        "2022040100" "2022041100" "2022042100" \
        "2022050100" "2022051100" "2022052100" \
        "2022060100" "2022061100" "2022062100" \
        # \
		# gfs part 3, ufs
        "2022070100" "2022071100" "2022072100" \
        "2022080100" "2022081100" "2022082100" \
        "2022090100" "2022091100" "2022092100" \
        "2022100100" "2022101100" "2022102100" \
        "2022110100" "2022111100" "2022112100" \
        "2022120100" "2022121100" "2022122100" \
        # \
		# gfs part 3, ufs, ifs
        "2024010100" "2024011100" "2024012100" \
        "2024020100" "2024021100" "2024022100" \
        "2024030100" "2024031100" "2024032100" \
        "2024040100" "2024041100" "2024042100" \
        "2024050100" "2024051100" "2024052100" \
        "2024060100" "2024061100" "2024062100" \
        "2024070100" "2024071100" "2024072100" \
        "2024080100" "2024081100" "2024082100" \
        "2024090100" "2024091100" "2024092100" \
        "2024100100" "2024101100" "2024102100" \
        "2024110100" "2024111100" "2024112100" \
        "2024120100" "2024121100" "2024122100" \
      )

EXCLUDE_NODES="u21g06"

for date in "${dates[@]}"; do
    echo ${case}, ${data}, ${ckpt}, ${date}
    sed -e "s|XXXXXX|${nlon}|g" \
        -e "s|YYYYYY|${nlat}|g" \
        -e "s|CCCCCC|${case}|g" \
        -e "s|TTTTTT|${data}|g" \
        -e "s|PPPPPP|${ckpt}|g" \
        -e "s|DDDDDD|${date}|g" \
        inference_scripts/run.sh > inference_scripts/run_${case}_${data}_${ckpt}_${date}.sh

    if [ -n "$EXCLUDE_NODES" ]; then
        sbatch --exclude=${EXCLUDE_NODES} inference_scripts/run_${case}_${data}_${ckpt}_${date}.sh
    else
        sbatch inference_scripts/run_${case}_${data}_${ckpt}_${date}.sh
    fi
done
