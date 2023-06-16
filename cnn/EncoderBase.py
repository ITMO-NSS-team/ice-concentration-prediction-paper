from typing import Optional
import torch.nn as nn
from EncoderBuilder import EncoderBuilder


class EncoderBase(nn.Module):
    def __init__(self):
        super(EncoderBase, self).__init__()
        self.convolved_size = None
        self.convolved_out_channels = None
        self.convolve = None
        self.to_vector = None
        self.transpose_input = None
        self.transpose = None

    def init_encoder(self,
                     input_size: list,
                     ts_number: int,
                     n_layers: int,
                     n_layers_to_transpose: Optional[int] = None,
                     convolve_params: Optional[dict] = None,
                     transpose_params: Optional[dict] = None):
        if not n_layers_to_transpose:
            n_layers_to_transpose = n_layers
        builder = EncoderBuilder(input_size, ts_number)
        if convolve_params:
            builder.default_convolve_params = convolve_params
        if transpose_params:
            builder.default_transpose_params = transpose_params
        self.convolve = builder.build_convolve_sequence(n_layers)
        self.to_vector = builder.build_to_vector_sequence()
        self.transpose_input = builder.build_transpose_input_sequence()
        self.transpose = builder.build_transpose_sequence(n_layers_to_transpose)
        self.convolved_out_channels = builder.convolved_out_channels
        self.convolved_size = builder.convolved_size

    def _resize_transpose_input(self, data):
        return data.view(-1, self.convolved_out_channels, self.convolved_size[0], self.convolved_size[1])

    def forward(self, x):
        x = self.convolve(x)
        x = self.to_vector(x)
        x = self.transpose_input(x)
        x = self._resize_transpose_input(x)
        x = self.transpose(x)
        return x

    def encode(self, x):
        x = self.convolve(x)
        x = self.to_vector(x)
        return x

    def decode(self, x):
        x = self.transpose_input(x)
        x = self._resize_transpose_input(x)
        x = self.transpose(x)
        return x
