import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from typing import Iterable
from profile_tools import ModelProfiler, DataLoaderGenerator
from DaYu.asyncPipelineModel import AsyncPipelineModel
import torch
from pprint import pprint
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(torch.cuda.is_available())


# args initialization
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="climax", help="")
parser.add_argument("--mode", type=str, default="multistream", help="eager, multistream")
parser.add_argument("--stream_num", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--batch_num", type=int, default=10)
parser.add_argument("--communication_time", type=bool, default=False)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--is_training", type=bool, default=False)
parser.add_argument("--batch_profile", type=bool, default=True)

args = parser.parse_args()

# model initialization
if args.model == "climax":
    from src.climax import get_model, get_inputs
elif args.model == "enformer":
    from src.enformer import get_model, get_inputs
else:
    raise ValueError(f"Model {args.model} not supported")

model = get_model()

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
        async_model.sliced_input = async_model._slice_input(data_loader)  # in multistream mode, inputs will be saved in the model.sliced_input
        profiler.model = async_model

    return profiler.compute_throughput(data_loader, batch_size=args.batch_size, mode=args.mode)


def batch_profile(args):
    batch_sizes = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    # batch_sizes = [1, 8, 16, 32]
    stream_nums = [1, 2, 4, 8, 16, 32, 64, 96, 108]
    # batch_sizes = [16]
    # stream_nums = [10]
    
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


def log_results(results):
    # category by batch_size and add {stream_num: (memory, throughput)}
    batch_size_results = {}
    stream_nums = []
    for result in results:
        batch_size = result["batch_size"]
        if batch_size not in batch_size_results:
            batch_size_results[batch_size] = {}
        stream_num = result["stream_num"]
        if stream_num not in batch_size_results[batch_size]:
            batch_size_results[batch_size][stream_num] = (result["memory"], result["throughput"])
        if stream_num not in stream_nums:
            stream_nums.append(stream_num)
    
    # log the results using table format but in txt file
    # write memory table first, using stream_num as row, batch_size as column
    batch_sizes = sorted(batch_size_results.keys())
    with open("./logs/batch_profile.txt", "w") as f:
        f.write("Memory Table\n")
        f.write("Stream Num\\Batch Size | " + " | ".join(map(str, batch_sizes)) + "\n")
        for stream_num in stream_nums:
            f.write(
                f"{stream_num} | " + 
                " | ".join(
                    f"{batch_size_results[batch_size][stream_num][0]:.3f}" 
                    if stream_num in batch_size_results[batch_size] and batch_size_results[batch_size][stream_num] 
                    else "None"
                    for batch_size in batch_sizes
                ) + "\n"
            )

        f.write("\nThroughput Table\n")
        f.write("Stream Num\\Batch Size | " + " | ".join(map(str, batch_sizes)) + "\n")
        for stream_num in stream_nums:
            f.write(
                f"{stream_num} | " + 
                " | ".join(
                    f"{batch_size_results[batch_size][stream_num][1]:.3f}" 
                    if stream_num in batch_size_results[batch_size] and batch_size_results[batch_size][stream_num] 
                    else "None"
                    for batch_size in batch_sizes
                ) + "\n"
            )   

if __name__ == "__main__":
    if args.batch_profile:
        results = batch_profile(args)
        log_results(results)
    else:
        single_profile(args, model)
    
    log_results(results)