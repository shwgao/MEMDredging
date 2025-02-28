import json
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from copy import deepcopy
import math
from torch.nn import init
from torch.utils.checkpoint import checkpoint_sequential

import os


def micro_batch(input, fn, batch_size, mini_batch, use_checkpoint=False):
    mini_batch_size = max(1, math.ceil(batch_size / mini_batch))
    output = []
    for i in range(mini_batch_size):
        start = i * mini_batch
        end = min((i+1) * mini_batch, batch_size)
        if use_checkpoint:
            x_i = checkpoint_sequential(fn, len(fn), input[start:end, :, :], use_reentrant=True)
        else:
            x_i = fn(input[start:end, :, :])    
        output.append(x_i)
    x = torch.cat(output, dim=0)
    return x


class Conv3DActMP(nn.Module):
    def __init__(
            self,
            conv_channel_in: int,
            conv_channel_out: int,
            isbn: bool = False,
    ):
        super().__init__()

        self.conv = nn.Conv3d(conv_channel_in, conv_channel_out, kernel_size=3, stride=1, padding=1, bias=True)
        self.isbn = isbn
        self.bn = nn.BatchNorm3d(conv_channel_out)
        self.act = nn.LeakyReLU(negative_slope=0.3)
        self.mp = nn.MaxPool3d(kernel_size=2, stride=2)

        torch.nn.init.xavier_uniform_(self.conv.weight)
        torch.nn.init.zeros_(self.conv.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.isbn:
            return self.mp(self.act(self.bn(self.conv(x))))
        else:
            return self.mp(self.act(self.conv(x)))


class CosmoFlow(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.conv_seq = nn.ModuleList()
        input_channel_size = 4

        for i in range(5):
            output_channel_size = 8 * (1 << i)
            self.conv_seq.append(Conv3DActMP(input_channel_size, output_channel_size))
            input_channel_size = output_channel_size
        
        self.conv_seq = nn.Sequential(*self.conv_seq)

        flatten_inputs = 128 // (2 ** 5)
        flatten_inputs = (flatten_inputs ** 3) * input_channel_size
        self.dense1 = nn.Linear(flatten_inputs, 128)
        self.dense2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 4)

        for layer in [self.dense1, self.dense2, self.output]:
            if hasattr(layer, 'weights'):
                torch.nn.init.xavier_uniform_(layer.weights)
                torch.nn.init.zeros_(layer.bias)

        self.layers = []
        for layer in self.conv_seq:
            self.layers.append(layer.conv)
        self.layers += [self.dense1, self.dense2, self.output]

        for layer in self.layers:
            if hasattr(layer, 'weights'):
                init.xavier_uniform_(layer.weights)
                
        self.batch_aggregate = False
        self.mini_batch = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_aggregate:
            x = micro_batch(x, self.conv_seq, x.shape[0], self.mini_batch)
        else:
            for i, conv_layer in enumerate(self.conv_seq):
                x = conv_layer(x)

        x = x.permute(0, 2, 3, 4, 1).flatten(1)

        x = nnf.leaky_relu(self.dense1(x.flatten(1)), negative_slope=0.3)
        x = nnf.leaky_relu(self.dense2(x), negative_slope=0.3)

        x = nnf.sigmoid(self.output(x)) * 1.2
        return x


def get_model():
    model = CosmoFlow()
    return model


def get_inputs(batch_size):
    x = torch.randn(batch_size, 4, 128, 128, 128)
    target = torch.randn(batch_size, 4)
    return (x, target), [0], True

class ModelWrapper(torch.nn.Module):
  # warp the loss computing and model forward pass
  def __init__(self):
    super().__init__()
    self.model = CosmoFlow()
    self.batch_aggregate = False
    self.mini_batch = 4
    
  def forward(self, inputs, target):
    self.model.batch_aggregate = self.batch_aggregate
    self.model.mini_batch = self.mini_batch
    outputs = self.model(inputs)
    loss = self.compute_loss(outputs, target)
    return loss
  
  def compute_loss(self, predicted, target):
    loss = nnf.mse_loss(predicted, target)
    return loss


def get_model():
    model = ModelWrapper()
    return model


if __name__ == "__main__":
    model = get_model().to("cuda")
    inputs, _, _ = get_inputs(10)
    loss = model(inputs[0].to("cuda"), inputs[1].to("cuda"))
    loss.backward()
    print(loss)
    print(torch.cuda.max_memory_allocated())
