import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.enformer import Enformer
from profiling.profile_tools import profile_with_torch, dump_snapshot, dump_onnx_graph, compute_throughput
from model_opt.apis import optimize
from DaYu.asyncPipelineModel import AsyncPipelineModel

model = Enformer()
model.to('cuda')
model.eval()

batch = 8

seq = torch.randint(0, 5, (batch, 196_608)).to('cuda')
inputs = seq

with torch.no_grad():
    output = model(inputs)
print(f"Output shape: {output.shape}")

# Profile model before optimization
if False:
    torch.cuda.empty_cache()
    print(f"Allocated memory before running: {torch.cuda.memory_allocated()/1024**3} GB")
    # profile_with_torch(model, inputs, f"enformer_before_opt_bz{batch}")
    # dump_snapshot(model, inputs, f"enformer_before_opt_bz{batch}")
    compute_throughput(model, inputs, 12*1024**3, mode='eager')

# Profile model after optimization
if False:
    # Optimize model
    model_opt = optimize(model, inputs, node_reordering=False)

    # Profile model after optimization
    torch.cuda.empty_cache()
    print(f"Allocated memory: {torch.cuda.memory_allocated()/1024**3} GB")
    profile_with_torch(model_opt, inputs, f"climax_after_opt_bz{batch}_no_reorder_no_optim")
    dump_snapshot(model_opt, inputs, f"climax_after_opt_bz{batch}_no_reorder_no_optim")

    result = model_opt(inputs)
    print(result.shape)

# profile model with multistream
if True:
    degree = 4
    model = AsyncPipelineModel(model, degree)
    
    torch.cuda.empty_cache()
    print(f"Allocated memory: {torch.cuda.memory_allocated()/1024**3} GB")
    # profile_with_torch(model, inputs, f"enformer_after_MS_bz{batch}_degree{degree}")
    # dump_snapshot(model, inputs, f"enformer_after_MS_bz{batch}_degree{degree}")
    compute_throughput(model, inputs, 12*1024**3, mode='multistream')

