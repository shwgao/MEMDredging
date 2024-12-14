import torch
import torch.nn as nn

class AsyncPipelineModel(nn.Module):
    def __init__(self, model, sliced_input=None, stream_num=2, device="cuda:0"):
        super(AsyncPipelineModel, self).__init__()
        self.model = model
        self.stream_num = stream_num
        self.streams = [torch.cuda.Stream() for _ in range(self.stream_num)]
        self.sliced_input = sliced_input
    
    def _slice_input(self, params):
        if type(params) == torch.Tensor:
            return torch.tensor_split(params, self.stream_num, 0)
        elif type(params) == tuple or type(params) == list:
            processed_params = []
            for param in params:
                if param is None:
                    processed_params.append(param)
                elif isinstance(param, torch.Tensor):
                    processed_params.append(torch.tensor_split(param, self.stream_num, 0))
                else:
                    # raise ValueError("The type is not supported now.")
                    processed_params.append([param for _ in range(self.stream_num)])

            sliced_params = [[] for _ in range(self.stream_num)]
            for pparams in processed_params:
                for i in range(self.stream_num):
                    sliced_params[i].append(pparams[i])
            return tuple(sliced_params)

        else:
            raise ValueError("The type is not supported now.")

    def _forward(self, x: tuple):
        for i in range(self.stream_num):
            with torch.cuda.stream(self.streams[i]):
                output = self.model(*x[i])
                yield output      
    
    # def forward(self, x):
    #     sliced_params = self._slice_input(x)
    #     print(sliced_params)
    #     for i in range(self.degree):
    #         with torch.cuda.stream(self.streams[i]):
    #             output = self.model(x[i])
    #             yield output
    
    def forward(self, x):
        if not self.sliced_input:
            self.sliced_input = self._slice_input(x)
    
        for i, result in enumerate(self._forward(self.sliced_input)):
            pass
        return result
    
