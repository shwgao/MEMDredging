import torch

def profile_with_torch(model, input, save_name, dump_snapshot=True, wait=1, warmup=1, active=3):
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
        
        if dump_snapshot:
            # clear memory
            torch.cuda.empty_cache()
            torch.cuda.memory._record_memory_history()
            results = model(*input)
            torch.cuda.memory._dump_snapshot(f"./logs/{save_name}/{save_name}.pickle")
                    