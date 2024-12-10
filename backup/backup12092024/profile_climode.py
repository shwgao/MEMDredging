import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import torch.nn.functional as F
from src.climode import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

number_year = 3
batch_size = 8

data = torch.rand(number_year, batch_size, 5, 32, 64).to(device)
past_sample = torch.rand(batch_size, 10, 32, 64).to(device)
const_channels_info = torch.rand(batch_size, 2, 32, 64).to(device)
lat_map = torch.rand(32, 64).to(device)
lon_map = torch.rand(32, 64).to(device)
time_steps = torch.arange(2, number_year).view(-1, 1).to(device)

step = 0

model = Climate_encoder_free_uncertain(5,2,out_types=5,method="euler",use_att=True,use_err=True,use_pos=False).to(device)
model.update_param([past_sample,const_channels_info.to(device),lat_map.to(device),lon_map.to(device)])
t = time_steps.float().to(device).flatten()
mean,std,_ = model(t,data)

exit()
with torch.no_grad():
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./profile/climODE/{profile_task}'),
    ) as p:
        for i in range(5):
            p.step()
            if step >=5:
                break
            model.update_param([past_sample,const_channels_info.to(device),lat_map.to(device),lon_map.to(device)])
            t = time_steps.float().to(device).flatten()
            mean,std,_ = model(t,data)
