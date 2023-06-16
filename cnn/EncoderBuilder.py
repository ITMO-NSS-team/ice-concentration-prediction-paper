import numpy as np
from typing import Optional

import torch.nn as nn


def calc_Conv2d_out(input_size: list,
                    padding: list,
                    dilation: list,
                    kernel_size: list,
                    stride: list):
    """
    :param input_size: Number of channels in the input image [h, w]
    :param padding: Padding added to all four sides of the input. Default: 0
    :param dilation: Spacing between kernel elements. Default: 1
    :param kernel_size: Size of the convolving kernel
    :param stride: Stride of the convolution. Default: 1
    :return: size of output image
    """
    h_out = (input_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
    w_out = (input_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
    return [int(h_out), int(w_out)]


def calc_ConvTranspose2d_out(input_size: list,
                             padding: list,
                             output_padding: list,
                             kernel_size: list,
                             stride: list,
                             dilation: str):
    """
    :param input_size: Number of channels in the input image [h, w]
    :param padding:  dilation * (kernel_size - 1) - padding zero-padding will be added to both sides
    of each dimension in the input. Default: 0
    :param output_padding: Additional size added to one side of each dimension in the output shape. Default: 0
    :param kernel_size: Size of the convolving kernel
    :param stride: Stride of the convolution. Default: 1
    :param dilation: Spacing between kernel elements. Default: 1
    :return: size of output image
    """
    h_out = (input_size[0] - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kernel_size[0] - 1) + output_padding[
        0] + 1
    w_out = (input_size[1] - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kernel_size[1] - 1) + output_padding[
        1] + 1
    return [int(h_out), int(w_out)]


class EncoderBuilder:
    def __init__(self, input_size: list, n_vectors: int):
        self.convolved_out_channels = None
        self.convolved_size = None
        self.input_size = input_size
        self.n_vectors = n_vectors
        self.default_convolve_params = {'padding': (0, 0),
                                        'kernel_size': (3, 3),
                                        'stride': (1, 1),
                                        'dilation': (1, 1)
                                        }
        self.default_transpose_params = {'padding': (0, 0),
                                         'output_padding': (0, 0),
                                         'kernel_size': (3, 3),
                                         'stride': (1, 1),
                                         'dilation': (1, 1),
                                         }
        self.activation_function = nn.ReLU()
        self.finish_activation_function = None

    def build_convolve_sequence(self, n_layers,
                                in_channels: Optional[int] = None,
                                params: Optional[dict] = None):
        input_layer_size = self.input_size
        if params:
            for key, value in params.items():
                self.default_convolve_params[key] = value

        params = self.default_convolve_params

        # число каналов в полтора раза меньше, чем изначальная картинка
        if not in_channels:
            in_channels = 0

        max_out_channels = ((input_layer_size[0] + input_layer_size[1]) * 0.5) // 2 + in_channels

        in_channels_list = list(np.arange(in_channels,
                                          max_out_channels - (max_out_channels - in_channels) // n_layers,
                                          (max_out_channels - in_channels) // n_layers))
        if in_channels == 0:
            in_channels_list[0] = 1

        out_channels = list(np.arange(in_channels_list[1],
                                      max_out_channels,
                                      (max_out_channels - in_channels) // n_layers))

        modules = []
        for n in range(n_layers):
            if input_layer_size[0] >= 5 and input_layer_size[1] >= 5:
                input_layer_size = calc_Conv2d_out(input_layer_size, **params)
                modules.append(nn.Conv2d(int(in_channels_list[n]), int(out_channels[n]), **params))
                modules.append(self.activation_function)
            else:
                print(f'Input size and parameters can not provide more than {n + 1} layers')
                break
        self.convolved_size = input_layer_size
        self.convolved_out_channels = int(out_channels[-1])
        return nn.Sequential(*modules)

    def build_to_vector_sequence(self):
        modules = [nn.AdaptiveAvgPool2d(output_size=1),
                   nn.Flatten(),
                   nn.Linear(self.convolved_out_channels,
                             self.n_vectors)]
        return nn.Sequential(*modules)

    def build_transpose_input_sequence(self):
        return nn.Linear(self.n_vectors, self.convolved_out_channels * self.convolved_size[0] * self.convolved_size[1])

    def build_transpose_sequence(self, n_layers,
                                 out_channels: Optional[int] = None,
                                 params: Optional[dict] = None):
        if params:
            for key, value in params.items():
                self.default_convolve_params[key] = value

        params = self.default_convolve_params

        if not out_channels:
            out_channels = 0
        out_channels_list = np.arange(out_channels,
                                      self.convolved_out_channels,
                                      (self.convolved_out_channels - out_channels) // n_layers)
        out_channels_list = out_channels_list[::-1]
        if out_channels == 0:
            out_channels_list[-1] = 1

        in_channels = np.arange(out_channels_list[-2],
                                self.convolved_out_channels + (self.convolved_out_channels - out_channels) // n_layers,
                                (self.convolved_out_channels - out_channels) // n_layers)
        in_channels = in_channels[::-1]
        in_channels[0] = self.convolved_out_channels

        modules = []
        while len(in_channels) > n_layers:
            in_channels = np.delete(in_channels, -1)
            out_channels_list = np.delete(out_channels_list, -2)
        for n in range(n_layers):
            modules.append(nn.ConvTranspose2d(in_channels[n], out_channels_list[n], **params))
            if self.finish_activation_function and n == range(n_layers)[-1]:
                modules.append(self.finish_activation_function)
            else:
                modules.append(self.activation_function)
        modules.append(nn.AdaptiveAvgPool2d(output_size=(self.input_size[0], self.input_size[1])))
        return nn.Sequential(*modules)
