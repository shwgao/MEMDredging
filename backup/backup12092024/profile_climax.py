import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from profile_tools import dump_snapshot, profile_with_torch, compute_throughput
from model_opt.apis import optimize


print(torch.__file__)
print(torch.__version__)
print(torch.version.cuda)
print(f"Cuda available: {torch.cuda.is_available()}")

from src.climax import ModelConfigGlobal, ClimaX

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

batch = 40
x = torch.randn(batch, 48, 32, 64, dtype=torch.float32).to(device)
lead_times = torch.tensor([72]*batch, dtype=torch.float32).to(device)
variables = model_config.default_vars
out_variables = model_config.out_variables

inputs = (x, None, lead_times, variables, out_variables, None, None)
with torch.no_grad():
    output = model(*inputs)
print(f"Output shape: {output.shape}")


# Profile model before optimization
if True:
    torch.cuda.empty_cache()
    print(f"Allocated memory: {torch.cuda.memory_allocated()/1024**3} GB")
    # profile_with_torch(model, inputs, f"climax_before_opt_bz{batch}")
    # dump_snapshot(model, inputs, f"climax_before_opt_bz{batch}")
    compute_throughput(model, inputs, 12*1024**3, mode='eager')

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