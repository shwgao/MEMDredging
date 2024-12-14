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
    # if args.batch_profile:
    #     results = batch_profile(args)
    #     log_results(results)
    # else:
    #     single_profile(args, model)
    results = [{'batch_size': 1,
        'memory': 1.0490388870239258,
        'stream_num': 1,
        'throughput': 40.075283636388036},
        {'batch_size': 8,
        'memory': 6.45582389831543,
        'stream_num': 1,
        'throughput': 48.96025664091559},
        {'batch_size': 8,
        'memory': 5.629185676574707,
        'stream_num': 2,
        'throughput': 49.54654939178258},
        {'batch_size': 8,
        'memory': 3.066067695617676,
        'stream_num': 4,
        'throughput': 48.87834724121796},
        {'batch_size': 8,
        'memory': 1.7147598266601562,
        'stream_num': 8,
        'throughput': 49.52550682435162},
        {'batch_size': 16,
        'memory': 12.61195182800293,
        'stream_num': 1,
        'throughput': 44.98665728529384},
        {'batch_size': 16,
        'memory': 10.94279956817627,
        'stream_num': 2,
        'throughput': 48.78650492141759},
        {'batch_size': 16,
        'memory': 5.782872200012207,
        'stream_num': 4,
        'throughput': 49.28356524963444},
        {'batch_size': 16,
        'memory': 3.251492500305176,
        'stream_num': 8,
        'throughput': 49.29286298222273},
        {'batch_size': 16,
        'memory': 1.8446426391601562,
        'stream_num': 16,
        'throughput': 50.087608987705394},
        {'batch_size': 32,
        'memory': 24.80518913269043,
        'stream_num': 1,
        'throughput': 44.576412615316585},
        {'batch_size': 32,
        'memory': 21.435139656066895,
        'stream_num': 2,
        'throughput': 44.59427450505942},
        {'batch_size': 32,
        'memory': 11.05180835723877,
        'stream_num': 4,
        'throughput': 48.80479051904985},
        {'batch_size': 32,
        'memory': 5.860142707824707,
        'stream_num': 8,
        'throughput': 50.37241661315135},
        {'batch_size': 32,
        'memory': 3.265286445617676,
        'stream_num': 16,
        'throughput': 49.23753421349734},
        {'batch_size': 32,
        'memory': 1.8505020141601562,
        'stream_num': 32,
        'throughput': 49.97663541201431},
        {'batch_size': 64,
        'memory': 24.95362663269043,
        'stream_num': 4,
        'throughput': 29.950509862689277},
        {'batch_size': 64,
        'memory': 11.06352710723877,
        'stream_num': 8,
        'throughput': 36.4470595906727},
        {'batch_size': 64,
        'memory': 5.871861457824707,
        'stream_num': 16,
        'throughput': 39.395207581653864},
        {'batch_size': 64,
        'memory': 3.277005195617676,
        'stream_num': 32,
        'throughput': 38.62997453723171},
        {'batch_size': 64,
        'memory': 1.8622207641601562,
        'stream_num': 64,
        'throughput': 49.975205325299804},
        {'batch_size': 128,
        'memory': 24.97706413269043,
        'stream_num': 8,
        'throughput': 27.934481926631882},
        {'batch_size': 128,
        'memory': 11.08696460723877,
        'stream_num': 16,
        'throughput': 35.493367063297036},
        {'batch_size': 128,
        'memory': 5.895298957824707,
        'stream_num': 32,
        'throughput': 39.43083346355659},
        {'batch_size': 128,
        'memory': 3.300442695617676,
        'stream_num': 64,
        'throughput': 34.31147556582868},
        {'batch_size': 128,
        'memory': 3.300442695617676,
        'stream_num': 96,
        'throughput': 39.898388054531225},
        {'batch_size': 128,
        'memory': 3.300442695617676,
        'stream_num': 108,
        'throughput': 41.919562362225165},
        {'batch_size': 256,
        'memory': 25.023939609527588,
        'stream_num': 16,
        'throughput': 27.750852594725462},
        {'batch_size': 256,
        'memory': 11.133840084075928,
        'stream_num': 32,
        'throughput': 34.34008568882589},
        {'batch_size': 256,
        'memory': 5.942174434661865,
        'stream_num': 64,
        'throughput': 38.79860891334042},
        {'batch_size': 256,
        'memory': 4.64474630355835,
        'stream_num': 96,
        'throughput': 37.04904649909449},
        {'batch_size': 256,
        'memory': 4.64474630355835,
        'stream_num': 108,
        'throughput': 39.233519180099904},
        {'batch_size': 512,
        'memory': 29.961440563201904,
        'stream_num': 32,
        'throughput': 28.19820156864974},
        {'batch_size': 512,
        'memory': 11.227591037750244,
        'stream_num': 64,
        'throughput': 34.611435787702284},
        {'batch_size': 512,
        'memory': 8.632612705230713,
        'stream_num': 96,
        'throughput': 36.296671678948144},
        {'batch_size': 512,
        'memory': 7.333841800689697,
        'stream_num': 108,
        'throughput': 36.33636503697724},
        {'batch_size': 1024,
        'memory': 30.148942470550537,
        'stream_num': 64,
        'throughput': 27.771246644676715},
        {'batch_size': 1024,
        'memory': 15.308842182159424,
        'stream_num': 96,
        'throughput': 32.33594927672947},
        {'batch_size': 1024,
        'memory': 14.010925769805908,
        'stream_num': 108,
        'throughput': 31.804695795711226},
        {'batch_size': 2048,
        'memory': 29.531758785247803,
        'stream_num': 108,
        'throughput': 27.04486672244616}]
    
    log_results(results)