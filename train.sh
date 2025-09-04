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
case="r0047" # same as r0044, but moisture_budget_correction off

#data="gfs"
data="ufs"

if [ ! -f "train_scripts/run_${case}.sh" ]; then
    sed -e "s|CCCCCC|${case}|g" \
        -e "s|DDDDDD|${data}|g" \
        train_scripts/run.sh > train_scripts/run_${case}.sh
fi

max_submissions=50
submit_count=0

# Desired number of GPU nodes for this run:
required_nodes=8
group_list=(u23 u22 u21 u20)

# Hardcoded list of nodes to avoid:
# nodelist subtracts from "scontrol show reservation | grep u2.g"
EXCLUDE_NODES="u21g10,u21g11,u21g12,u21g13,u21g14,u22g01,u22g02,u22g08"

find_nodelist() {
    sinfo_out=$(sinfo -p u1-h100,u1-mi300x,u1-gh -N -o "%N %T" | awk '$2=="idle" {print $1}')

    declare -A node_groups
    for prefix in "${group_list[@]}"; do
        node_groups[$prefix]=""
    done

    for node in $sinfo_out; do
        if echo "$EXCLUDE_NODES" | grep -qw "$node"; then
            continue
        fi
        for prefix in "${group_list[@]}"; do
            if [[ $node == ${prefix}g* ]]; then
                node_groups[$prefix]="${node_groups[$prefix]} $node"
            fi
        done
    done

    check_combination() {
        local nodes=()
        for g in "$@"; do
            nodes+=(${node_groups[$g]})
        done
        if [ ${#nodes[@]} -ge $required_nodes ]; then
            echo "${nodes[@]:0:$required_nodes}"
            return 0
        fi
        return 1
    }

    for g1 in "${group_list[@]}"; do
        result=$(check_combination $g1) && echo "$result" && return 0
    done

    for i in "${!group_list[@]}"; do
        for j in $(seq $((i+1)) $((${#group_list[@]}-1))); do
            result=$(check_combination ${group_list[$i]} ${group_list[$j]}) && echo "$result" && return 0
        done
    done

    for i in "${!group_list[@]}"; do
        for j in $(seq $((i+1)) $((${#group_list[@]}-1))); do
            for k in $(seq $((j+1)) $((${#group_list[@]}-1))); do
                result=$(check_combination ${group_list[$i]} ${group_list[$j]} ${group_list[$k]}) && echo "$result" && return 0
            done
        done
    done

    result=$(check_combination "${group_list[@]}") && echo "$result" && return 0
    return 1
}

while [ $submit_count -lt $max_submissions ]; do
    queue=$(squeue -u "$USER" --Format="Name:100" --noheader | grep -w "train_ACE_${case}")

    if [ -z "$queue" ]; then
        echo "[$(date)] Job train_ACE_${case} is not in queue. Submitting job #$((submit_count + 1))..."

        NODELIST=$(find_nodelist)
        if [ -n "$NODELIST" ]; then
            echo "[$(date)] Found preferred nodes: $NODELIST"
            sbatch --nodelist=$(echo "$NODELIST" | tr ' ' ',') "train_scripts/run_${case}.sh"
        else
            echo "[$(date)] No preferred nodelist found. Letting Slurm scheduler decide (but excluding explicitly specified nodes)."
            sbatch --exclude=$EXCLUDE_NODES "train_scripts/run_${case}.sh"
        fi

        ((submit_count++))
    else
        echo "[$(date)] Job train_ACE_${case} is still in the queue. Waiting..."
    fi

    sleep 600
done

echo "[$(date)] Reached the maximum number of submissions: $max_submissions. Exiting."
