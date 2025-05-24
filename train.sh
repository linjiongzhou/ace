#!/bin/bash

#case="r0000" # First run
#case="r0001" # First 2 GPUs run
#case="r0002" # Try 4 GPUs in one node
#case="r0003" # Try 4 GPUs in two nodes
case="r0004" # add new training data (h500, TMP850, ...)

sed -e "s|CCCCCC|${case}|g" \
    train_scripts/run.sh > train_scripts/run_${case}.sh
sbatch train_scripts/run_${case}.sh
