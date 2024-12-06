import os
import sys
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.climax import ModelConfigGlobal, ClimaX
from asyncPipelineModel import AsyncPipelineModel
from torch.profiler import profile, ProfilerActivity
from profiling.profile_tools import profile_with_torch

# 2. Model setup
model_config = ModelConfigGlobal()
model = ClimaX(
    default_vars=model_config.default_vars,
    img_size=model_config.img_size,
    patch_size=model_config.patch_size,
    embed_dim=model_config.embed_dim,
    depth=model_config.depth,
    decoder_depth=model_config.decoder_depth,
    num_heads=model_config.num_heads,
    mlp_ratio=model_config.mlp_ratio,
    drop_path=model_config.drop_path,
    drop_rate=model_config.drop_rate,
)

# 3. Device setup and model transfer
device = torch.device("cuda:0")
model = model.to(device)
model.eval()

# 4. Input preparation
batch = 30
x = torch.randn(batch, 48, 32, 64, dtype=torch.float32).to(device)
lead_times = torch.tensor([72]*batch, dtype=torch.float32).to(device)
variables = model_config.default_vars
out_variables = model_config.out_variables

# 5. Model inference
inputs = (x, None, lead_times, variables, out_variables, None, None)

degree = 10
asyn_model = AsyncPipelineModel(model, degree)
x_ = torch.chunk(x, degree, 0)
lead_times_ = torch.chunk(lead_times, degree, 0)

profile_with_torch(asyn_model, inputs, f"climax_SL_asyn_model_bz{batch}")
