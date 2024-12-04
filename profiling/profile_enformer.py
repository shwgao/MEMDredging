import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.enformer import Enformer
from profiling.profile_tools import profile_with_torch, dump_snapshot, dump_onnx_graph

model = Enformer()
model.to('cuda')
model.eval()

seq = torch.randint(0, 5, (8, 196_608)).to('cuda')
inputs = (seq,)

with torch.no_grad():
    output = model(*inputs)
print(f"Output shape: {output.shape}")

# traced_model = torch.fx.symbolic_trace(model)
traced_model = torch.fx.TracedModule(model)
model = torch.fx.GraphModule(model, traced_model.graph)
print(model.code)

# # 6. Profile model
# profile_with_torch(model, inputs, "enformer_inference")
# dump_snapshot(model, inputs, "enformer_inference")
# dump_onnx_graph(model, inputs, "enformer_inference")

