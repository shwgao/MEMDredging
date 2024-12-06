import torch
import torch.nn as nn

class AsyncPipelineModel(nn.Module):
    def __init__(self, model, degree=2):
        super(AsyncPipelineModel, self).__init__()
        self.model = model
        self.degree = degree
        self.streams = [torch.cuda.Stream() for _ in range(self.degree)]
    
    def _slice_input(self, params):
        if type(params) == torch.Tensor:
            return torch.chunk(params, self.degree, 0)
        elif type(params) == tuple:
            processed_params = []
            for param in params:
                if param is None:
                    processed_params.append(param)
                elif param is torch.Tensor:
                    processed_params.append(torch.chunk(params, self.degree, 0))
                else:
                    raise ValueError("The type is not supported now.")

            sliced_params = [[]] * self.degree
            for pparams in processed_params:
                for i in range(self.degree):
                    sliced_params[i].append(pparams[i])
            return tuple(sliced_params)
                
        else:
            raise ValueError("The type is not supported now.")

    def _forward(self, x: tuple):
        for i in range(self.degree):
            with torch.cuda.stream(self.streams[i]):
                output = self.model(*x[i])
                yield output      
        

    def forward(self, x):
        sliced_params = self._slice_input(x)
        print(sliced_params)
        for i in range(self.degree):
            with torch.cuda.stream(self.streams[i]):
                output = self.model(x[i])
                yield output
        