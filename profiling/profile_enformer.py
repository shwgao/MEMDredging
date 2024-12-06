import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.enformer import Enformer
from profiling.profile_tools import profile_with_torch, dump_snapshot, dump_onnx_graph

model = Enformer()
model.to('cuda')
model.eval()

batch = 8

seq = torch.randint(0, 5, (batch, 196_608)).to('cuda')
inputs = (seq,)

with torch.no_grad():
    output = model(*inputs)
print(f"Output shape: {output.shape}")

# Profile model before optimization
if True:
    torch.cuda.empty_cache()
    print(f"Allocated memory before running: {torch.cuda.memory_allocated()/1024**3} GB")
    profile_with_torch(model, inputs, f"enformer_before_opt_bz{batch}")
    dump_snapshot(model, inputs, f"enformer_before_opt_bz{batch}")

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

