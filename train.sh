#!/bin/bash

#case="r0000" # First run
#case="r0001" # First 2 GPUs run
#case="r0002" # Try 4 GPUs in one node
#case="r0003" # Try 4 GPUs in two nodes
#case="r0004" # same as r0001, add new training data (h500, TMP850, ...)
#case="r0005" # same as r0004, fix extrapolation
case="r0006" # same as r0005, fix solar incident

sed -e "s|CCCCCC|${case}|g" \
    train_scripts/run.sh > train_scripts/run_${case}.sh
sbatch train_scripts/run_${case}.sh
