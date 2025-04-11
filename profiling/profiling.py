import os
import sys
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from typing import Iterable
from profile_tools2 import ModelProfiler, DataLoaderGenerator
from DaYu.asyncPipelineModel import AsyncPipelineModel
from utils import log_results, overwrite_dir, clean_cuda_cache, get_model_parameters
import torch
import shutil
from pprint import pprint
import torchvision
# from torch.export import export
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print('torch.cuda.is_available():', torch.cuda.is_available())
torch.manual_seed(42)

torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True
# import torch._dynamo as dynamo
# dynamo.config.cache_size_limit = 64


def single_profile(args, model):    
    # args.batch_size = args.batch_size // 2
    inputs, batch_index, is_batched = get_inputs(args.batch_size)
    
    # TODO: we have to add communication time for comprehensive comparison
    if args.communication_time:
        dlg = DataLoaderGenerator(inputs, args.batch_size, args.batch_num, batch_index, 
                                  is_batched=is_batched, prefetch_factor=2, num_workers=1)
        data_loader = dlg.get_dataloader()
    else:
        data_loader = [i.to(args.device) if hasattr(i, "to") else i for i in inputs]
    
    num_params, memory_size = get_model_parameters(model)
    print(f"Model parameters: {num_params}, {memory_size} MB")
    input_memory = sum(i.numel() * 4 / 1024**2 for i in inputs if hasattr(i, "numel"))
    print(f"Input memory: {input_memory} MB")
    
    if args.is_training:
        model.train()
    else:
        model.eval()
     
    # profiler
    if args.rockmate:
        profiler = ModelProfiler(model, device=args.device, is_training=args.is_training,
                                 offloading=args.offload_optimizer, rockmate=args.rockmate, input_=data_loader)
    else:
        profiler = ModelProfiler(model, device=args.device, is_training=args.is_training,
                                 offloading=args.offload_optimizer)

    # multistream mode
    if args.mode == "multistream":
        async_model = AsyncPipelineModel(model, stream_num=args.stream_num)
        # in multistream mode, inputs will be saved in the model.sliced_input
        async_model.sliced_input = async_model._slice_input(data_loader, batch_index)  
        profiler.model = async_model
    else:
        if isinstance(data_loader, list):
            data_loader = tuple(data_loader)

    if args.dump_snapshot:
        profiler.dump_snapshot(data_loader, args.model)
        
    clean_cuda_cache()

    if args.torch_profiling:
        save_name = 'logs/' + args.model + '/' + args.mode + f'train_{args.is_training}-bz{args.batch_size}-{args.hardware}-bagg_{args.batch_aggregate}-mb_{args.mini_batch}-check_{args.checkpointing}-off_{args.offload_optimizer}'
        overwrite_dir(save_name)
        profiler.torch_profiling(data_loader, save_name, wait=1, warmup=1, active=3)
    
    clean_cuda_cache()
    
    if args.is_training:
        return profiler.compute_throughput(data_loader, batch_size=args.batch_size, mode=args.mode)
    else:
        with torch.no_grad():
            return profiler.compute_throughput(data_loader, batch_size=args.batch_size, mode=args.mode)


def batch_profile(args, model, batch_sizes, stream_nums): 
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
            
            clean_cuda_cache()
            
    pprint(results)
    return results


def batch_profile_mini_batch(args, model, batch_size, stream_nums): 
    results = []
    mini_batch_sizes = list(range(1, batch_size, 2))
    for mini_batch_size in mini_batch_sizes:
        model.mini_batch = mini_batch_size
        args.mini_batch = mini_batch_size
        for stream_num in stream_nums:
            print(f"--------------------batch_size: {batch_size}, mini_batch_size: {mini_batch_size}, stream_num: {stream_num}--------------------")
            if stream_num > batch_size:
                continue
            try:
                args.stream_num = stream_num
                result = single_profile(args, model)
                result["batch_size"] = mini_batch_size
                result["stream_num"] = stream_num
                results.append(result)
            except Exception as e:
                print(f"Error: {e}: batch_size={batch_size}, mini_batch_size={mini_batch_size}, stream_num={stream_num}")
                continue
            
            clean_cuda_cache()
            
    pprint(results)
    return results


def check_gradients(args, model):
    profiler = ModelProfiler(model, device=args.device, is_training=args.is_training)
    model2 = copy.deepcopy(model)
    profiler2 = ModelProfiler(model2, device=args.device, is_training=args.is_training)
    
    profiler.check_parameters_close(model, model2)
    profiler2.model.batch_aggregate = True
    profiler2.model.mini_batch = 8
    
    inputs, batch_index, is_batched = get_inputs(args.batch_size)
    
    data_loader = [i.to(args.device) if hasattr(i, "to") else i for i in inputs]
    for i in range(3):
        output1 = profiler._compute(data_loader)
        output2 = profiler2._compute(data_loader)
        profiler.check_output_close(output1, output2)
    
    profiler.check_gradients_close(model, model2)


# args initialization
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="climax", help="")
parser.add_argument("--mode", type=str, default="eager", help="eager, multistream")
parser.add_argument("--stream_num", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--batch_num", type=int, default=10)
parser.add_argument("--communication_time", type=bool, default=False)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--is_training", type=bool, default=True)
parser.add_argument("--batch_profile", type=bool, default=False)
parser.add_argument("--dump_snapshot", type=bool, default=False)
parser.add_argument("--torch_profiling", type=bool, default=False)
parser.add_argument("--backend", type=str, default="pytorch", help="pytorch, no_caching, cuda")
parser.add_argument("--hardware", type=str, default="V100", help="V100, A100")
parser.add_argument("--batch_cat_aggregate", type=bool, default=False, help="Only useful for climax")
parser.add_argument("--batch_aggregate", action="store_true", help="Enable batch aggregation")
parser.add_argument("--no-batch_aggregate", dest="batch_aggregate", action="store_false", help="Disable batch aggregation")
parser.add_argument("--mini_batch", type=int, default=8)
parser.add_argument("--checkpointing", action="store_true", help="Enable checkpointing")
parser.add_argument("--no-checkpointing", dest="checkpointing", action="store_false", help="Disable checkpointing")
parser.add_argument("--offload_optimizer", action="store_true", help="Enable offloading optimizer")
parser.add_argument("--no-offload_optimizer", dest="offload_optimizer", action="store_false", help="Disable offloading optimizer")
parser.add_argument("--rockmate", action="store_true", help="Enable rockmate")
parser.add_argument("--no-rockmate", dest="rockmate", action="store_false", help="Disable rockmate")
parser.add_argument("--torch_compile", action="store_true", help="Enable torch compile")
parser.add_argument("--no-torch_compile", dest="torch_compile", action="store_false", help="Disable torch compile")
parser.set_defaults(batch_aggregate=False)
parser.set_defaults(checkpointing=False)
parser.set_defaults(offload_optimizer=False)
parser.set_defaults(rockmate=False)
parser.set_defaults(torch_compile=False)

args = parser.parse_args()

# model initialization
if args.model == "climax":
    from src.climax import get_model, get_inputs
    batch_sizes = list(range(1, 60, 2))
    args.batch_size = 54#32 if args.is_training else 46
    args.mini_batch = 8
elif args.model == "enformer":
    from src.enformer import get_model, get_inputs
    batch_sizes = list(range(1, 12, 1))
    args.batch_size = 3#2 if args.is_training else 10
    args.mini_batch = 1
elif args.model == "climode": # seems not suitable for our work because of large portion of cpus computation
    batch_sizes = list(range(2, 100, 4))
    from src.climode import get_model, get_inputs
    args.batch_size = 10
elif args.model == "cosmoflow":
    from src.cosmoflow import get_model, get_inputs
    batch_sizes = list(range(1, 60, 2))
    args.batch_size = 156#96 if args.is_training else 128
    args.mini_batch = 16
elif args.model == "sam":
    from src.sam import get_model, get_inputs
    batch_sizes = list(range(1, 20, 1))
    args.batch_size = 5# 3 if args.is_training else 3
    args.mini_batch = 2
elif args.model == "simmim":
    from src.simmim import get_model, get_inputs
    batch_sizes = list(range(1, 30, 1))
    args.batch_size = 7#5 if args.is_training else 10
    args.mini_batch = 2
else:
    raise ValueError(f"Model {args.model} not supported")

# if args.backend == "no_caching":
#     os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
# elif args.backend == "cuda":
#     os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'backend:cudaMallocAsync'

if __name__ == "__main__":
    if args.batch_profile:
        from utils import read_from_file, plot_data_twinx

        model = get_model()
        model.batch_cat_aggregate = args.batch_cat_aggregate
        model.batch_aggregate = args.batch_aggregate
        model.mini_batch = args.mini_batch
        model.checkpointing = args.checkpointing
        if args.torch_compile:
            model = torch.compile(model)
        results = batch_profile(args, model, batch_sizes, [1])
        results = batch_profile_mini_batch(args, model, args.batch_size, [1])
        save_name = f"{args.model}/micro_batch"
        file_name = log_results(results, save_name)
        memory_table, throughput_table, batch_sizes, stream_nums = read_from_file(file_name)
        plot_data_twinx(memory_table, throughput_table, stream_nums, batch_sizes, 
        save_name=save_name, x_axis="batch")
    else:
        print(f"Running with batch_aggregate={args.batch_aggregate}, checkpointing={args.checkpointing}, offload_optimizer={args.offload_optimizer}")
        
        model = get_model()
        model.batch_cat_aggregate = args.batch_cat_aggregate
        model.batch_aggregate = args.batch_aggregate
        model.mini_batch = args.mini_batch
        model.checkpointing = args.checkpointing
        if args.torch_compile:
            model = torch.compile(model)
        single_profile(args, model)
