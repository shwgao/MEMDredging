import json
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from copy import deepcopy
from torch.nn import init

import os


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    return (x,), [0], True

if __name__ == "__main__":
    input = torch.randn(1, 4, 128, 128, 128)
    model = CosmoFlow()
    output = model(input)
    print(output.shape)
