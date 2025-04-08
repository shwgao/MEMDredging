#!/bin/bash

# 定义模型数组
models=("climax" "enformer" "cosmoflow" "sam" "simmim")
mini_batch_sizes=1to10
# models=("climax")

# 创建或清空output.txt文件
> output_mini_batch.txt

# 循环遍历每个模型
for model in "${models[@]}"
do
    echo "Testing model: $model" >> output_mini_batch.txt
    echo "===================" >> output_mini_batch.txt
    
    # 配置4: batch_aggregate=true, checkpointing=true, offload_optimizer=false, rockmate=true
    echo "Config 4: batch_aggregate=false, checkpointing=false, offload_optimizer=false, rockmate=true" >> output_mini_batch.txt
    python profiling/profiling.py --model "$model" --no-batch_aggregate --no-checkpointing --no-offload_optimizer --rockmate >> output_mini_batch.txt 2>&1
    echo "" >> output_mini_batch.txt
    echo "-------------------" >> output_mini_batch.txt
done

echo "Testing completed" >> output.txt