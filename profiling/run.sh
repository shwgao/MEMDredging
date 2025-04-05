#!/bin/bash

# 定义模型数组
# models=("climax" "enformer" "cosmoflow" "sam" "simmim")
models=("climax")

# 创建或清空output.txt文件
> output.txt

# 循环遍历每个模型
for model in "${models[@]}"
do
    echo "Testing model: $model" >> output.txt
    echo "===================" >> output.txt
    
    # 配置1: batch_aggregate=false, checkpointing=false, offload_optimizer=false
    echo "Config 1: batch_aggregate=false, checkpointing=false, offload_optimizer=false" >> output.txt
    python profiling/profiling.py --model "$model" --no-batch_aggregate --no-checkpointing --no-offload_optimizer >> output.txt 2>&1
    echo "" >> output.txt
    echo "-------------------" >> output.txt
    
    # 配置2: batch_aggregate=false, checkpointing=false, offload_optimizer=true
    echo "Config 2: batch_aggregate=false, checkpointing=false, offload_optimizer=true" >> output.txt
    python profiling/profiling.py --model "$model" --no-batch_aggregate --no-checkpointing --offload_optimizer >> output.txt 2>&1
    echo "" >> output.txt
    echo "-------------------" >> output.txt
    
    # 配置3: batch_aggregate=false, checkpointing=true, offload_optimizer=false
    echo "Config 3: batch_aggregate=false, checkpointing=true, offload_optimizer=false" >> output.txt
    python profiling/profiling.py --model "$model" --no-batch_aggregate --checkpointing --no-offload_optimizer >> output.txt 2>&1
    echo "" >> output.txt
    echo "-------------------" >> output.txt
    
    # 配置4: batch_aggregate=true, checkpointing=true, offload_optimizer=false
    echo "Config 4: batch_aggregate=true, checkpointing=true, offload_optimizer=false" >> output.txt
    python profiling/profiling.py --model "$model" --batch_aggregate --checkpointing --no-offload_optimizer >> output.txt 2>&1
    echo "" >> output.txt
    echo "-------------------" >> output.txt
done

echo "Testing completed" >> output.txt