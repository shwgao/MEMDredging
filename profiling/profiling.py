import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from typing import Iterable
from profile_tools import ModelProfiler, DataLoaderGenerator
from DaYu.asyncPipelineModel import AsyncPipelineModel
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(torch.cuda.is_available())


# args initialization
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="climax", help="")
parser.add_argument("--mode", type=str, default="multistream", help="eager, multistream")
parser.add_argument("--stream_num", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--batch_num", type=int, default=10)
parser.add_argument("--communication_time", type=bool, default=False)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--is_training", type=bool, default=False)

args = parser.parse_args()


# model initialization
if args.model == "climax":
    from src.climax import get_model_and_data
    model, inputs, batch_index, is_batched = get_model_and_data(args.batch_size)
elif args.model == "enformer":
    from src.enformer import get_model_and_data
    model, inputs, batch_index, is_batched = get_model_and_data()
else:
    raise ValueError(f"Model {args.model} not supported")

# profiler
profiler = ModelProfiler(model, device=args.device, is_training=args.is_training)

# TODO: we have to add communication time for comprehensive comparison
if args.communication_time:
    dlg = DataLoaderGenerator(inputs, args.batch_size, args.batch_num, batch_index, is_batched=is_batched)
    data_loader = dlg.get_dataloader()
else:
    data_loader = [i.to(args.device) if hasattr(i, "to") else i for i in inputs]

# multistream mode
if args.mode == "multistream":
    model = AsyncPipelineModel(model, stream_num=args.stream_num)
    model.sliced_input = model._slice_input(data_loader)  # in multistream mode, inputs will be saved in the model.sliced_input
    profiler.model = model

profiler.compute_throughput(data_loader, batch_size=args.batch_size, mode=args.mode)
