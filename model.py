import torch
import torch.nn as nn
import numpy as np


class TaskNet(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        '''
        input_dim:输入特征的维度
        hidden_dim:隐藏层维度
        query_dim:query向量的维度,因为其直接由隐藏层输出,故query_dim = output_dim
        此网络模块的最终输出是query向量
        '''
        super(TaskNet, self).__init__()
        self.query_dim = hidden_dim
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, X):
        query = self.fc1(X)
        return query


class Expert(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        '''
        input_dim:输入特征的维度
        hidden_dim:隐藏层维度
        key_dim:key向量的维度,因为其直接由隐藏层输出,故key_dim = output_dim
        此网络模块的最终输出是key向量
        '''
        super(Expert, self).__init__()
        self.key_dim = hidden_dim
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, X):
        key = self.fc1(X)
        return key


class Tower(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super(Tower, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, X):
        out = self.fc1(X)
        return out


class AOE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task_num, expert_num):
        super(AOE, self).__init__()
        # some dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.task_num = task_num
        self.expert_num = expert_num
        # model
        self.tasknet_list = nn.ModuleList([TaskNet(self.input_dim, self.hidden_dim) for i in range(self.task_num)])
        self.expert_list = nn.ModuleList([Expert(self.input_dim, self.hidden_dim) for i in range(self.expert_num)])
        self.tower_list = nn.ModuleList([Tower(self.hidden_dim, self.output_dim) for i in range(self.task_num)])
        # vector
        self.query = []
        self.key = []
        self.value = []
        self.alpha = []
        self.attention = []
        # function
        self.softmax = nn.Softmax(dim=0)  # dim=0代表作用于行向量，dim=1代表作用于列向量

    def forward(self, X):

        # 计算query向量
        self.query = [task_net(X) for task_net in self.tasknet_list]
        self.query = torch.stack(self.query)

        # 计算key向量
        self.key = [expert(X) for expert in self.expert_list]
        self.key = torch.stack(self.key)  # shape:torch.Size([5, 64])

        # 计算权重alpha向量
        self.alpha = []
        for q in self.query:
            alpha_temp = []
            for k in self.key:
                cos_similarity = torch.cosine_similarity(q, k).mean() # 采用余弦相似度计算行向量之间的相似度（得到54个值， 然后取平均）
                alpha_temp.append(cos_similarity)
            alpha_temp = torch.stack(alpha_temp)
            self.alpha.append(alpha_temp)
        self.alpha = torch.stack(self.alpha)
        self.alpha = self.softmax(self.alpha)    

        # 计算value向量
        self.value = self.key.clone()

        # 计算attention值
        self.attention = []
        for a_i in self.alpha:
            # 计算第一个任务关于专家网络的attention值
            temp = []
            for a_ij, v_i in zip(a_i, self.value):
                # 权重和value值分别相乘, 存入列表后求和
                attention_i = a_ij*v_i
                temp.append(attention_i)

            self.attention.append(sum(temp))
        self.attention = torch.stack(self.attention)      # shape:torch.Size([4, 54, 64])                                       

        # 拼接attention值和TaskNet输出值，然后送入Tower
        # 其实TaskNet的输出值就是query
        query_add_attention = self.query + self.attention   # shape:torch.Size([4, 54, 64])
  
        ## 传入TOWER
        tower_input = query_add_attention
        tower_output = [tower(ti) for tower, ti in zip(self.tower_list, tower_input)]


        return tower_output
