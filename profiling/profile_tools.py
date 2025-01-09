import torch
import time
import numpy as np
import os
from torch.utils.data import DataLoader
from DaYu.asyncPipelineModel import AsyncPipelineModel
class ModelProfiler:
    def __init__(self, model=None, save_dir='./logs', device='cuda', is_training=False):
        self.model = model
        self.save_dir = save_dir
        self._wrapped_model = None
        self.device = device
        self.model.to(self.device)
        self.is_training = is_training
    
    @property
    def wrapped_model(self):
        if self._wrapped_model is None and not isinstance(self.model, AsyncPipelineModel):
            self._wrapped_model = ModelWrapper(self.model)
        elif isinstance(self.model, AsyncPipelineModel):
            self._wrapped_model = self.model
        return self._wrapped_model

    def torch_profiling(self, input_data, save_name, wait=1, warmup=1, active=3):
        with torch.no_grad():
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
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(f'{self.save_dir}/{save_name}-{time.strftime("%Y-%m-%d-%H-%M-%S")}'),
                ) as p:
                    for i in range(10):
                        p.step()
                        if i >= wait + warmup + active:
                            break
                        self.model(input_data)

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
            return self._compute_eager_throughput(data_loader, batch_size, warmup, iter)
        elif mode == 'multistream':
            return self._compute_multistream_throughput(data_loader, batch_size, warmup, iter)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _compute_eager_throughput(self, data_loader, batch_size, warmup=1, iter=5):
        time_list = []
        memory_list = []
        # Create CUDA events for timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        if isinstance(data_loader, DataLoader):
            for _ in range(iter):
                torch.cuda.synchronize()
                start_event.record()
                
                for batch in data_loader:
                    input_data = batch[0]
                    self.model(*input_data)
                
                end_event.record()
                memory_list.append(torch.cuda.max_memory_reserved() / 1024**3)
                torch.cuda.synchronize()
                time_list.append(start_event.elapsed_time(end_event) / 1000.0)
            batch_size = sum([len(batch[0]) for batch in data_loader])
        else:
            with torch.no_grad():
                for _ in range(warmup):
                    self.model(*data_loader)
                
                for _ in range(iter):
                    # torch.cuda.empty_cache()
                    # torch.cuda.synchronize()
                    start_event.record()
                    
                    self.model(*data_loader)
                    end_event.record()
                    
                    # print(f"memory: {torch.cuda.max_memory_allocated() / 1024**3}")    
                    memory_list.append(torch.cuda.max_memory_reserved() / 1024**3)
                    torch.cuda.synchronize()
                    time_list.append(start_event.elapsed_time(end_event) / 1000.0)

        return self._calculate_statistics(time_list, batch_size, memory_list)

    def _compute_multistream_throughput(self, input_data, batch_size, warmup=2, iter=5):
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
    
    