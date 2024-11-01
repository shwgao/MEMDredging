import torch
import time
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # Unpack the tuple of inputs and pass them to the model
        return self.model(*x)


def profile_with_torch(model, input, save_name, wait=1, warmup=1, active=3):
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
                on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./logs/{save_name}'),
            ) as p:
                for i in range(10):
                    p.step()
                    if i >=5:
                        break
                    
                    results = model(*input)


def dump_snapshot(model, input, save_name):
    with torch.no_grad():
        # clear memory
        results = model(*input)
        print(f"Results shape: {results.shape}")
        
        torch.cuda.empty_cache()
        torch.cuda.memory._record_memory_history()
        results = model(*input)
        torch.cuda.memory._dump_snapshot(f"./logs/{save_name}/{save_name}.pickle")



def dump_onnx_graph(model, input, save_name):
    wrapped_model = ModelWrapper(model)
    
    # move everything to cpu avoiding different devices issues
    wrapped_model = wrapped_model.to("cpu")
    input = [i.to("cpu") if hasattr(i, "to") else i for i in input]
    
    with torch.no_grad():
        wrapped_model(input)
        torch.onnx.export(wrapped_model, input, f"./logs/{save_name}/{save_name}.onnx")
        

if __name__ == "__main__":
    # saving 20GB memory but not using it
    saving_memory = 12
    # Each float32 element takes 4 bytes
    bytes_per_element = 4  # float32
    target_bytes = saving_memory * 1024 * 1024 * 1024  # 20GB in bytes
    num_elements = target_bytes // bytes_per_element

    # Create tensor and move to GPU
    # Using float32 as default dtype
    large_tensor = torch.ones(num_elements, device='cuda')
    
    while True:
        # print memory usage every minute
        print(f"Memory usage: {torch.cuda.memory_allocated(large_tensor.device) / 1024**2} MB")
        time.sleep(60) 