from typing import Optional

import torch
import torch.nn as nn
from EncoderBuilder import EncoderBuilder


class EncoderForecasterBase(nn.Module):
    def __init__(self):
        super(EncoderForecasterBase, self).__init__()
        self.convolved_size = None
        self.convolved_out_channels = None
        self.convolve = None
        self.to_vector = None
        self.transpose_input = None
        self.transpose = None

    def init_encoder(self,
                     input_size: list,
                     n_layers: int,
                     in_channels: Optional[int] = None,
                     out_channels: Optional[int] = None,
                     n_layers_to_transpose: Optional[int] = None,
                     convolve_params: Optional[dict] = None,
                     transpose_params: Optional[dict] = None,
                     finish_activation_function: Optional = None,
                     add_branch: Optional = 1):
        if not n_layers_to_transpose:
            n_layers_to_transpose = n_layers
        builder = EncoderBuilder(input_size, 0)
        builder.finish_activation_function = finish_activation_function
        self.convolve = builder.build_convolve_sequence(n_layers, in_channels, convolve_params)
        builder.convolved_out_channels = builder.convolved_out_channels * add_branch
        self.transpose = builder.build_transpose_sequence(n_layers_to_transpose, out_channels, transpose_params)
        self.convolved_size = builder.convolved_size

    def forward(self, x, x1=None):
        x = self.convolve(x)
        if x1 is not None:
            x1 = self.convolve(x1)
            #x = torch.cat((x, x1), dim=1)
            x = torch.cat((x, x1), dim=0)
        x = self.transpose(x)
        return x
