import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from DaYu.asyncPipelineModel import AsyncPipelineModel
from torch.profiler import ExecutionTraceObserver

from pynvml import nvmlInit, nvmlDeviceGetMemoryInfo, nvmlDeviceGetHandleByIndex


class ModelProfiler:
    def __init__(self, model=None, save_dir='./logs', device='cuda', is_training=False):
        self.model = model
        self.save_dir = save_dir
        self._wrapped_model = None
        self.device = device
        self.model.to(self.device)
        self.is_training = is_training
        self.optimizer = None
        
        if self.is_training:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        nvmlInit()  # Add NVML initialization
    
    @property
    def wrapped_model(self):
        if self._wrapped_model is None and not isinstance(self.model, AsyncPipelineModel):
            self._wrapped_model = ModelWrapper(self.model)
        elif isinstance(self.model, AsyncPipelineModel):
            self._wrapped_model = self.model
        return self._wrapped_model
    
    @staticmethod
    def check_gradients_close(model1, model2):
        index = 0
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            if param1.grad is None and param2.grad is None:
                print(f"Warning: Gradient is None for one or both parameters")
                continue
            if not torch.allclose(param1.grad, param2.grad, atol=1e-6, rtol=1e-6):
                print(f"Gradient mismatch for parameter {index}, Gradient difference (max): {torch.max(torch.abs(param1.grad - param2.grad))}")
            # else:
            #     print(f"Gradient for parameter {index} has shape {param1.grad.shape} and {param2.grad.shape} and is close")
            index += 1
        return True
    
    @staticmethod
    def check_parameters_close(model1, model2):
        index = 0
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            if not torch.allclose(param1, param2, atol=1e-8, rtol=1e-8):
                print(f"Parameter mismatch for parameter {index}, Parameter difference (max): {torch.max(torch.abs(param1 - param2))}")
            # else:
            #     print(f"Parameter for parameter {index} has shape {param1.shape} and {param2.shape} and is close")
            index += 1
        return True
    
    def check_output_close(self, output1, output2):
        if not torch.allclose(output1, output2, atol=1e-8, rtol=1e-8):
            print(f"Output mismatch, Output difference (max): {torch.max(torch.abs(output1 - output2))}")
        return True
    
    def torch_profiling(self, input_data, save_name, wait=1, warmup=1, active=3):
        with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_modules=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f'{save_name}/tensorboard.pt.trace.json'),
                execution_trace_observer=(
                    ExecutionTraceObserver().register_callback(f"{save_name}/execution_trace.json")
                ),
            ) as p:
                if not self.is_training:
                    with torch.no_grad():
                        for i in range(wait + warmup + active):
                            self._compute(input_data)
                            p.step()
                else:
                    for i in range(wait + warmup + active):
                        self._compute(input_data)
                        p.step()

    def dump_snapshot(self, input_data, save_name):
        if not os.path.exists(f"{self.save_dir}/{save_name}"):
            os.makedirs(f"{self.save_dir}/{save_name}")
        with torch.no_grad():
            torch.cuda.empty_cache()
            torch.cuda.memory._record_memory_history()
            self.wrapped_model(input_data)
            torch.cuda.memory._dump_snapshot(f"{self.save_dir}/{save_name}/{save_name}-{time.strftime('%Y-%m-%d-%H-%M-%S')}.pickle")

    def dump_onnx_graph(self, input_data, save_name):
        # move everything to cpu avoiding different devices issues
        cpu_model = self.wrapped_model.to("cpu")
        cpu_input = [i.to("cpu") if hasattr(i, "to") else i for i in input_data]
        
        with torch.no_grad():
            cpu_model(cpu_input)
            torch.onnx.export(cpu_model, cpu_input, f"{self.save_dir}/{save_name}/{save_name}.onnx")

    def compute_throughput(self, data_loader, batch_size, mode='eager', warmup=2, iter=5):
        if mode == 'eager':
            return self.compute_eager_throughput(data_loader, batch_size, warmup, iter)
        elif mode == 'multistream':
            return self.compute_multistream_throughput(data_loader, batch_size, warmup, iter)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _compute(self, data):
        if self.is_training:
            self.optimizer.zero_grad()
            
        loss = self.model(*data)
        
        if self.is_training:
            loss.backward()
            self.optimizer.step()
        
        return loss
    
    def compute_eager_batched_data(self, data_loader, start_event, end_event, warmup=1, iter=5):
        time_list = []
        memory_list = []
        for _ in range(warmup):
            self._compute(data_loader)
        
        for _ in range(iter):
            # torch.cuda.reset_peak_memory_stats()
            start_event.record()
            
            self._compute(data_loader)
            
            end_event.record()
            
            memory_list.append(torch.cuda.max_memory_allocated() / 1024**3)
            # handle = nvmlDeviceGetHandleByIndex(0)
            # memory_list.append(nvmlDeviceGetMemoryInfo(handle).used / 1024**3)
            torch.cuda.synchronize()
            time_list.append(start_event.elapsed_time(end_event) / 1000.0)
            
        return time_list, memory_list

    def compute_eager_dataloader(self, data_loader, start_event, end_event, warmup=1, iter=5):
        time_list = []
        memory_list = []
        
        count = 0
        for batch in data_loader:
            input_data = batch[0]
            input_data = tuple(
                i.to(self.device) if isinstance(i, torch.Tensor) else i
                for i in input_data
            )
            self._compute(input_data)
            
            count += 1
            if count >= warmup:
                break
        
        count = 0
        for batch in data_loader:
            start_event.record()
            input_data = batch[0]
            input_data = tuple(
                i.to(self.device) if isinstance(i, torch.Tensor) else i
                for i in input_data
            )
            self._compute(input_data)
            
            end_event.record()
            torch.cuda.synchronize()
            memory_list.append(torch.cuda.max_memory_reserved() / 1024**3)
            time_list.append(start_event.elapsed_time(end_event) / 1000.0)
            
            count += 1
            if count >= iter:
                break
            
        return time_list, memory_list
    
    def compute_eager_throughput(self, data_loader, batch_size, warmup=1, iter=5):
        time_list = []
        memory_list = []
        # Create CUDA events for timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        if isinstance(data_loader, DataLoader):
            time_list, memory_list = self.compute_eager_dataloader(data_loader, start_event, end_event, warmup, iter)
        else:
            time_list, memory_list = self.compute_eager_batched_data(data_loader, start_event, end_event, warmup, iter)
            
        return self._calculate_statistics(time_list, batch_size, memory_list)

    def compute_multistream_throughput(self, input_data, batch_size, warmup=2, iter=5):
        time_list = []
        memory_list = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            for _ in range(warmup):
                self.model(input_data)
                
            for _ in range(iter):
                
                torch.cuda.synchronize()
                # start = time.time()
                start_event.record()
                start_event.synchronize()
                
                self.model(input_data)
                memory_list.append(torch.cuda.max_memory_reserved() / 1024**3)
                
                torch.cuda.synchronize()
                end_event.record()
                end_event.synchronize()
                time_list.append(start_event.elapsed_time(end_event) / 1000.0)
                # end = time.time()
                # print(f"Time: {end - start:.3f} seconds")

        return self._calculate_statistics(time_list, batch_size, memory_list)

    def _calculate_statistics(self, time_list, batch_size, memory_list):
        time_array = np.array(time_list)
        mean_time = np.mean(time_array)
        std_time = np.std(time_array)
        mean_memory = np.mean(memory_list)
        std_memory = np.std(memory_list)
        throughput = batch_size / mean_time
        min_throughput = batch_size / time_array.max()
        max_throughput = batch_size / time_array.min()

        print(f"Mean time/batch:    {mean_time:.3f}         seconds")
        print(f"Time  Std:          {std_time:.3f}         seconds")
        print(f"Memory usage:       {mean_memory:.4g}          GB")
        print(f"Memory std:         {std_memory:.4g}             GB")
        print(f"Throughput:         {throughput:.4g}({min_throughput:.4g}~{max_throughput:.4g}) samples/second")

        return {
            'mean_time_per_batch': mean_time,
            'batch_size': batch_size,
            'memory': mean_memory,
            'throughput': throughput
        }



class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return self.model(*x)


class DataLoaderGenerator:
    def __init__(self, example_data, batch_size, batch_num, batch_index, is_batched=False, **dataloader_kwargs):
        self.example_data = example_data
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.batch_index = batch_index
        self.is_batched = is_batched
        self.dataloader_kwargs = dataloader_kwargs

    def get_dataloader(self):
        if self.is_batched:
            batched_data = [self.example_data] * self.batch_num
        else:
            batched_data = []
            for _ in range(self.batch_num):
                one_batch = list(self.example_data)
            
            for data_index in self.batch_index:
                if isinstance(one_batch[data_index], torch.Tensor):
                    one_batch[data_index] = one_batch[data_index].expand(self.batch_size, *one_batch[data_index].shape[1:])
            
            batched_data.append(tuple(one_batch))
        
        def custom_collate(batch):
            return batch
        
        return DataLoader(batched_data, batch_size=1, collate_fn=custom_collate, **self.dataloader_kwargs)


if __name__ == "__main__":
    # test the dataloader generator
    # create example data
    x = torch.randn(10, 48, 32, 64, dtype=torch.float32)
    lead_times = torch.tensor([72]*10, dtype=torch.float32)
    inputs = (x, None, lead_times)
    batch_size = 10
    batch_num = 10
    batch_index = [0]
    dataloader_kwargs = {'pin_memory': True, 'num_workers': 0}
    dataloader_generator = DataLoaderGenerator(inputs, batch_size, batch_num, batch_index, is_batched=True, **dataloader_kwargs)
    dataloader = dataloader_generator.get_dataloader()
    for batch in dataloader:
        print(len(*batch))
    
    