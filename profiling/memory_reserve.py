import torch
from time import sleep
reserve_memory = 20

big_tensor = torch.ones(reserve_memory * 1024**3//4, dtype=torch.float32).to('cuda')

print(f"Memory usage of big_tensor: {big_tensor.element_size() * big_tensor.nelement() / 1024**3} GB")

while True:
    sleep(10)
    print(f"Memory usage of big_tensor: {big_tensor.element_size() * big_tensor.nelement() / 1024**3} GB")