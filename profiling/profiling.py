import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from typing import Iterable
from profile_tools import ModelProfiler, DataLoaderGenerator
from DaYu.asyncPipelineModel import AsyncPipelineModel
from utils import log_results
import torch
from pprint import pprint
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print('torch.cuda.is_available():', torch.cuda.is_available())


def single_profile(args, model):    
    inputs, batch_index, is_batched = get_inputs(args.batch_size)
    
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
        async_model = AsyncPipelineModel(model, stream_num=args.stream_num)
        # in multistream mode, inputs will be saved in the model.sliced_input
        async_model.sliced_input = async_model._slice_input(data_loader, batch_index)  
        profiler.model = async_model
    else:
        data_loader = tuple(data_loader)

    if args.dump_snapshot:
        profiler.dump_snapshot(data_loader, args.model)

    if args.torch_profiling:
        profiler.torch_profiling(data_loader, args.model)

    return profiler.compute_throughput(data_loader, batch_size=args.batch_size, mode=args.mode)


def batch_profile(args, model):
    # batch_sizes = list(range(2, 50, 4))
    batch_sizes = list(range(2, 33, 2))
    # batch_sizes = [4]
    stream_nums = [1]
    
    results = []
    for batch_size in batch_sizes:
        args.batch_size = batch_size
        for stream_num in stream_nums:
            print(f"--------------------batch_size: {batch_size}, stream_num: {stream_num}--------------------")
            if stream_num > batch_size:
                continue
            try:
                args.stream_num = stream_num
                result = single_profile(args, model)
                result["batch_size"] = batch_size
                result["stream_num"] = stream_num
                results.append(result)
            except Exception as e:
                print(f"Error: {e}: batch_size={batch_size}, stream_num={stream_num}")
                continue
            
            # clear cuda cache
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**3:.4g} GB")
            
    pprint(results)
    return results


# args initialization
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="enformer", help="")
parser.add_argument("--mode", type=str, default="multistream", help="eager, multistream")
parser.add_argument("--stream_num", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--batch_num", type=int, default=10)
parser.add_argument("--communication_time", type=bool, default=False)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--is_training", type=bool, default=False)
parser.add_argument("--batch_profile", type=bool, default=True)
parser.add_argument("--dump_snapshot", type=bool, default=False)
parser.add_argument("--torch_profiling", type=bool, default=False)

args = parser.parse_args()

# model initialization
if args.model == "climax":
    from src.climax import get_model, get_inputs
elif args.model == "enformer":
    from src.enformer import get_model, get_inputs
elif args.model == "climode":
    from src.climode import get_model, get_inputs
else:
    raise ValueError(f"Model {args.model} not supported")


if __name__ == "__main__":    
    if args.batch_profile:
        model = get_model()
        model = torch.compile(model)
        results = batch_profile(args, model)
        log_results(results, args.model)
    else:
        model = get_model()
        model = torch.compile(model)
        single_profile(args, model)
