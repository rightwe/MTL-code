import os

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from PIL import Image
from exp_utils import read_data, draw_model, exp, exp_batch
import json

dataloader = read_data(pic_path='../Data/First_EXP/hwt_pictures.csv',
                       original_path='../Data/First_EXP/NIMS_Fatigue.csv')
target_names = ['Fatigue', 'Tensile', 'Fracture', 'Hardness']


class scaled_dot_product_attention(nn.Module):
    '''
    Scaled 指的是 Q和K计算得到的相似度 再经过了一定的量化，具体就是 除以 根号下K_dim;
    Dot-Product 指的是 Q和K之间 通过计算点积作为相似度.
    '''
    def __init__(self, att_dropout=0.0):
        super(scaled_dot_product_attention, self).__init__()
        self.dropout = nn.Dropout(att_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None):
        '''
        args:
            q: [batch_size, q_length, q_dimension]
            k: [batch_size, k_length, k_dimension]
            v: [batch_size, v_length, v_dimension]
            q_dimension = k_dimension = v_dimension
            scale: 缩放因子
        return:
            context, attention
        '''
        # 快使用神奇的爱因斯坦求和约定吧！
        attention = torch.einsum('ijk,ilk->ijl', [q, k])# query和key向量相乘
        if scale:
            attention = attention * scale
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.einsum('ijl,ilk->ijk', [attention, v])
        return context, attention


class multi_heads_self_attention(nn.Module):
    '''
    所谓的self-attention就是Q,K,V都是一样的
    '''
    def __init__(self, feature_dim, num_heads=1, dropout=0.0):
        super(multi_heads_self_attention, self).__init__()

        self.dim_per_head = feature_dim // num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(feature_dim, self.dim_per_head * num_heads)
        self.linear_k = nn.Linear(feature_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(feature_dim, self.dim_per_head * num_heads)

        self.sdp_attention = scaled_dot_product_attention(dropout) # 实例化attention模块
        self.linear_attention = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_dim)
        # self.linear_1 = nn.Linear(feature_dim, 256)
        # self.linear_2 = nn.Linear(256, feature_dim)
        # self.layer_final = nn.Linear(feature_dim, 3)

    def forward(self, key, value, query):
        residual = query
        batch_size = key.size(0)

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * self.num_heads, -1, self.dim_per_head)
        value = value.view(batch_size * self.num_heads, -1, self.dim_per_head)
        query = query.view(batch_size * self.num_heads, -1, self.dim_per_head)

        if key.size(-1) // self.num_heads != 0:
            scale = (key.size(-1) // self.num_heads) ** -0.5
        else:
            scale = 1
        context, attention = self.sdp_attention(query, key, value, scale)

        # concat heads
        context = context.view(
            batch_size, -1, self.dim_per_head * self.num_heads)

        output = self.linear_attention(context)
        output = self.dropout(output)
        output = torch.squeeze(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        # # pass through linear
        # output = nn.functional.relu(self.linear_1(output))
        # output = nn.functional.relu(self.linear_2(output))

        # # pass through layer final
        # output = self.layer_final(output)

        return output, attention


class multimodel(nn.Module):
    def __init__(self):
        super(multimodel, self).__init__()
        # x: (1, 1, 24, 21)
        self.pic_block = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, stride=3),  # (1, 1, 8, 7)
            # nn.BatchNorm2d(3),
            # nn.ReLU(inplace=True),
            nn.Conv2d(3, 8, kernel_size=3, stride=1),  # (1, 1, 6, 5)
            # nn.BatchNorm2d(3),
            # nn.ReLU(inplace=True),
            nn.Conv2d(8, 4, kernel_size=3, stride=1),  # (1, 1, 4, 3)
            # nn.BatchNorm2d(3),
            # nn.ReLU(inplace=True),
            nn.Conv2d(4, 1, kernel_size=2, stride=1),  # (1, 1, 3, 2)
            # nn.BatchNorm2d(1),
            # nn.ReLU(inplace=True),
        )
        self.att1 = multi_heads_self_attention(feature_dim=16, num_heads=2)
        self.linear_1 = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
        )
        self.linear_2 = nn.Sequential(
            nn.Linear(6, 4),
            nn.ReLU(inplace=True),
        )
        self.att2 = multi_heads_self_attention(feature_dim=12, num_heads=2)
        self.output = nn.Sequential(
            nn.BatchNorm1d(12),
            nn.Linear(12, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, pics, other):
        out_1, _ = self.att1(other, other, other)
        out_1 = self.linear_1(out_1)
        out_pic = self.pic_block(pics)
        out_2 = self.linear_2(torch.flatten(out_pic, start_dim=1))

        out = torch.cat([out_1, out_2], dim=1)
        out_3, _ = self.att2(out, out, out)
        out = self.output(out_3)
        return out
