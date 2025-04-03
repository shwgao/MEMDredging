import torch
import torch.nn as nn
import torch.optim as optim

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.attention = nn.MultiheadAttention(64, 8)  # Fixed typo in variable name
        self.fc1 = nn.Linear(64, 10)

    def forward(self, x):
        # Ensure input is in the correct shape for Conv2d: [batch_size, channels, height, width]
        x = self.conv1(x)  # x should be [B, 3, H, W]
        batch_size = x.size(0)
        # Reshape for attention: [seq_len, batch_size, embed_dim]
        x = x.flatten(2).permute(2, 0, 1)
        x, _ = self.attention(x, x, x)
        # Reshape back and apply final linear layer
        x = x.permute(1, 2, 0).mean(dim=-1)
        x = self.fc1(x)
        return x

    def get_random_input(self, batch_size, height, width):
        # Generate correct input shape for Conv2d: [batch_size, channels, height, width]
        return torch.randn(batch_size, 3, height, width)


def torch_profiling(model, input_data, save_name='toy_model', wait=1, warmup=1, active=3):
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
            # execution_trace_observer=(
            #     ExecutionTraceObserver().register_callback(f"{save_name}/execution_trace.json")
            # ),
        ) as p:
            for i in range(wait + warmup + active):
                model(input_data)
                p.step()


def main():
    model = ToyModel().to('cuda')
    input = model.get_random_input(1, 64, 64).to('cuda')  # [1, 3, 32, 32]
    
    torch_profiling(model, input, save_name='toy_model', wait=1, warmup=1, active=3)

if __name__ == "__main__":
    main()