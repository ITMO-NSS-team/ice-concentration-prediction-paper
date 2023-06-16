import os
import numpy as np
from torch.utils.data import TensorDataset
import torch


def single_output_tensor(forecast_len, images_array, additional_images_arrays):
    y = images_array[forecast_len:]
    x = images_array[:-forecast_len]
    for array in additional_images_arrays:
        additional_matrix = array[forecast_len]
        x = np.vstack((x, additional_matrix[None]))
    tensor_x = torch.Tensor(x)
    tensor_y = torch.Tensor(y)
    dataset = TensorDataset(tensor_x, tensor_y)
    return dataset


def multioutput_tensor(pre_history_len, forecast_len, images_array, additional_images=None):
    """
    :param pre_history_len: длина предыстории
    :param forecast_len: длина предсказания
    :param images_array: np.array с временным рядом из матриц
    :return: TensorDataset с набором предикторов и таргетов
    """
    x_train = []
    x1_train = []
    y_train = []
    for i in range(images_array.shape[0] - forecast_len - pre_history_len):
        x = images_array[i:i + pre_history_len, :, :]
        x_train.append(x)
        if additional_images is not None:
            x1 = images_array[i:i + pre_history_len, :, :]
            x1_train.append(x1)
        y = images_array[i + pre_history_len:i + pre_history_len + forecast_len, :, :]
        y_train.append(y)
    y_train = np.array(y_train)
    x_train = np.array(x_train)
    x1_train = np.array(x1_train)
    tensor_x = torch.Tensor(x_train)
    tensor_x1 = torch.Tensor(x1_train)
    tensor_y = torch.Tensor(y_train)
    if additional_images is not None:
        dataset = TensorDataset(tensor_x, tensor_x1, tensor_y)
    else:
        dataset = TensorDataset(tensor_x, tensor_y)
    return dataset


def multioutput_binary_output_tensor(pre_history_len, forecast_len, images_array, threshold, x_transform=True):
    """
    :param threshold: порог для присвоения бинарных меток
    :param pre_history_len: длина предыстории
    :param forecast_len: длина предсказания
    :param images_array: np.array с временным рядом из матриц
    :return: TensorDataset с набором предикторов и таргетов
    """
    x_train = []
    y_train = []
    for i in range(images_array.shape[0] - forecast_len - pre_history_len):
        x = images_array[i:i + pre_history_len, :, :]
        if x_transform:
            x[x > threshold] = 1
            x[x <= threshold] = 0
        x_train.append(x)
        y = images_array[i + pre_history_len:i + pre_history_len + forecast_len, :, :]
        y[y > threshold] = 1
        y[y <= threshold] = 0
        y_train.append(y)
    y_train = np.array(y_train)
    x_train = np.array(x_train)
    tensor_x = torch.Tensor(x_train)
    tensor_y = torch.Tensor(y_train)
    dataset = TensorDataset(tensor_x, tensor_y)
    return dataset

def multioutput_binary_output_tensor_masie(pre_history_len, forecast_len, images_array):
    """
    :param threshold: порог для присвоения бинарных меток
    :param pre_history_len: длина предыстории
    :param forecast_len: длина предсказания
    :param images_array: np.array с временным рядом из матриц
    :return: TensorDataset с набором предикторов и таргетов
    """
    x_train = []
    y_train = []
    for i in range(images_array.shape[0] - forecast_len - pre_history_len):
        x = images_array[i:i + pre_history_len, :, :]
        x_train.append(x)
        y = images_array[i + pre_history_len:i + pre_history_len + forecast_len, :, :]
        y_train.append(y)
    y_train = np.array(y_train)
    x_train = np.array(x_train)
    tensor_x = torch.Tensor(x_train)
    tensor_y = torch.Tensor(y_train)
    dataset = TensorDataset(tensor_x, tensor_y)
    return dataset

def multioutput_tensor_with_additional_predictors(pre_history_len, forecast_len, images_array,
                                                  additional_images_arrays,
                                                  pre_history_for_additional_images):
    """
    :param pre_history_for_additional_images: список с int количеством предыстории для каждого из доп параметров
    :param additional_images_arrays: список дополнительных np.array с временными рядами
    :param pre_history_len: длина предыстории
    :param forecast_len: длина предсказания
    :param images_array: np.array с временным рядом из матриц
    :return: TensorDataset с набором предикторов и таргетов
    """
    x_train = []
    y_train = []
    for i in range(images_array.shape[0] - forecast_len - pre_history_len):
        x = images_array[i:i + pre_history_len, :, :]
        for array, pre_hist_size in zip(additional_images_arrays, pre_history_for_additional_images):
            additional_matrix = array[i + pre_history_len-pre_hist_size:i + pre_history_len]
            x = np.vstack((x, additional_matrix))
        x_train.append(x)
        y = images_array[i + pre_history_len:i + pre_history_len + forecast_len, :, :]
        y_train.append(y)
    y_train = np.array(y_train)
    x_train = np.array(x_train)
    tensor_x = torch.Tensor(x_train)
    tensor_y = torch.Tensor(y_train)
    dataset = TensorDataset(tensor_x, tensor_y)
    return dataset
