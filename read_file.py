import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def read_file(filepath='./4tasks-encode.xlsx', iScaler=False):
    data = pd.read_excel(filepath).values
    data_X, data_y = data[:, 0:18], data[:, 18:22]
    if iScaler:
        scaler = StandardScaler()
        data_X = scaler.fit_transform(data_X)
    return data_X, data_y


def getTensorDataset(my_x, my_y):
    tensor_x = torch.Tensor(my_x)
    tensor_y = torch.Tensor(my_y).long()
    return torch.utils.data.TensorDataset(tensor_x, tensor_y)
