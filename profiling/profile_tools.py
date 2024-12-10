import torch
import time
import numpy as np
from torch.utils.data import DataLoader

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
        if self._wrapped_model is None:
            self._wrapped_model = ModelWrapper(self.model)
        return self._wrapped_model

    def profile_with_torch(self, input_data, save_name, wait=1, warmup=1, active=3):
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
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(f'{self.save_dir}/{save_name}'),
                ) as p:
                    for i in range(10):
                        p.step()
                        if i >= 5:
                            break
                        self.model(input_data)

    def dump_snapshot(self, input_data, save_name):
        with torch.no_grad():
            torch.cuda.empty_cache()
            torch.cuda.memory._record_memory_history()
            self.model(input_data)
            torch.cuda.memory._dump_snapshot(f"{self.save_dir}/{save_name}/{save_name}.pickle")

    def dump_onnx_graph(self, input_data, save_name):
        # move everything to cpu avoiding different devices issues
        cpu_model = self.wrapped_model.to("cpu")
        cpu_input = [i.to("cpu") if hasattr(i, "to") else i for i in input_data]
        
        with torch.no_grad():
            cpu_model(cpu_input)
            torch.onnx.export(cpu_model, cpu_input, f"{self.save_dir}/{save_name}/{save_name}.onnx")

    def compute_throughput(self, data_loader, batch_size, mode='eager'):
        if mode == 'eager':
            return self._compute_eager_throughput(data_loader, batch_size)
        elif mode == 'multistream':
            return self._compute_multistream_throughput(data_loader)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _compute_eager_throughput(self, data_loader, batch_size):
        time_list = []
        for _ in range(5):
            if isinstance(data_loader, DataLoader):
                # TODO: compute the troughput with communication time
                start_time = time.time()
                for batch in data_loader:
                    input_data = batch[0]
                    self.model(*input_data)
            else:
                start_time = time.time()
                self.model(*data_loader)
            time_list.append(time.time() - start_time)

        return self._calculate_statistics(time_list, len(data_loader)*batch_size)

    def _compute_multistream_throughput(self, input_data):
        time_list = []
        for test_iter in range(10):
            start_time = time.time()
            self.model(input_data)
            time_list.append(time.time() - start_time)

        return self._calculate_statistics(time_list, len(input_data))

    def _calculate_statistics(self, time_list, num_samples):
        time_array = np.array(time_list)
        mean_time = np.mean(time_array)
        std_time = np.std(time_array)
        throughput = num_samples / mean_time

        print(f"Mean time: {mean_time:.4f} seconds")
        print(f"Memory usage: {torch.cuda.memory_allocated() / 1024**3} GB")
        print(f"Standard deviation: {std_time:.4f} seconds")
        print(f"Throughput: {throughput:.2f} samples/second")

        return {
            'mean_time': mean_time,
            'std_time': std_time,
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
    
    