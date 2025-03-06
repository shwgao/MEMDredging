#!/bin/bash

# List of models to test
models=("climax" "enformer" "cosmoflow")

# Loop through each model
for model in "${models[@]}"; do
    # Run with batch_aggregate=True
    python profiling/profiling.py --model $model --batch_aggregate True
    
    # Run with batch_aggregate=False
    python profiling/profiling.py --model $model --batch_aggregate False
done