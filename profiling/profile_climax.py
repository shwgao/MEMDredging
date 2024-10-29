import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time

# 1. Environment checks
print(torch.__file__)
print(torch.__version__)
print(torch.version.cuda)
print(f"Cuda available: {torch.cuda.is_available()}")

from src.climax import ModelConfigGlobal, ClimaX

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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# 4. Input preparation
batch = 40
x = torch.randn(batch, 48, 32, 64, dtype=torch.float32).to(device)
lead_times = torch.tensor([72]*batch, dtype=torch.float32).to(device)
variables = model_config.default_vars
out_variables = model_config.out_variables

# 5. Model inference
inputs = (x, None, lead_times, variables, out_variables, None, None)
with torch.no_grad():
    output = model(*inputs)
print(f"Output shape: {output.shape}")


# --------------------------------------------------------------------------
# OLLA optimization
# --------------------------------------------------------------------------

# olla.optimize(model, input)
# importer = olla.torch.torch_graph_importer.TorchGraphImporter()
# (
#     g,
#     pytorch_node_order,
#     fx_graph,
#     fx_to_df_map,
# ) = importer.import_via_fx(
#     model,
#     input,
#     mode="eval",
#     cleanup=True,
#     treat_output_as_fake=False,
#     return_node_ordering=True,
#     return_fx_graph=True,
# )
# assert(g.is_valid())
# g.canonicalize()
# g.constrain_weight_updates()
# g.constrain_tensor_generators()
# assert(g.is_valid())

# node_order = str([node for node in fx_graph.graph.nodes])

# g.dump("orig_graph_climax", format="png")