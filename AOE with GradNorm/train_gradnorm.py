'''
train.py里的训练函数太辣鸡了
此处对其进行大规模优化

'''

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from read_file import getTensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 十折交叉验证

loss_fn = nn.MSELoss()


def train(lr=1e-2, n_epochs=1000, loss_fn=loss_fn, seed=1, X=None, y=None, model=None, isGradNorm=True):

    optimizer = optim.Adam(model.parameters(), lr=lr) # optimizer一定要放在循环里，因为优化器是和参数绑定的
    X_train, X_test, y_train, y_test = train_test_split(X.astype(np.float32), y, test_size=0.33, random_state=seed)

    ### 这两行构建数据集的代码可以放到main函数里
    train_loader = DataLoader(dataset=getTensorDataset(X_train, y_train), batch_size=54)
    val_loader = DataLoader(dataset=getTensorDataset(X_test,y_test),batch_size=37)

    # Training loop
    for epoch in range(n_epochs):
        model.train()


        r2_t1_val, r2_t2_val, r2_t3_val, r2_t4_val = 0,0,0,0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device).to(torch.float32)
            y_batch = y_batch.to(device).to(torch.float32)
            yhat = model(x_batch)
   
            # 对每个tower分别计算train_r2
            r2_train = [r2_score(y_true.detach().cpu().numpy(), y_hat.to(torch.float32).detach().cpu().numpy()) for y_true, y_hat in zip([y_batch[:,i] for i in range(y_batch.shape[1])], yhat)]

            # 对每个tower分别计算损失，一共有4个
            label_loss_train = [loss_fn(y_hat.to(torch.float32), y_true.view(-1, 1)) for y_true, y_hat in zip([y_batch[:,i] for i in range(y_batch.shape[1])], yhat)]

            # 计算总损失
            loss_weighted = torch.mul(torch.stack(label_loss_train), model.dynamic_weights).sum()

            optimizer.zero_grad()
            loss_weighted.backward(retain_graph=True)


            #--------------------------梯度规范操作------------------------------------------
            if isGradNorm:
                # 1. 获取共享层最后一层的神经网络参数
                last_layer_weight = model.expert_list[1].last_layer ## 此代码是伪代码，最后再将其实现

                # 2. 初始化0时刻的任务损失
                if epoch == 0:
                    # set L(0)
                    if torch.cuda.is_available():
                        initial_task_loss = loss_weighted.data.cpu()
                    else:
                        initial_task_loss = loss_weighted.data
                    initial_task_loss = initial_task_loss.numpy()

                # 3. 计算逆损失率
                if torch.cuda.is_available():
                    loss_ratio = sum(label_loss_train).data.cpu().numpy() / initial_task_loss
                else:
                    loss_ratio = sum(label_loss_train).data.numpy() / initial_task_loss
                # r_i(t)
                inverse_train_rate = loss_ratio / np.mean(loss_ratio)
                
                # 4. 计算每个任务的label损失在W上的梯度W_grad
                #    然后再用这个梯度与该任务对应的动态权重相乘----> 然后进行范数计算，得到了梯度范数
                norms = []
                for i in range(len(label_loss_train)):
                    W_grad = torch.autograd.grad(outputs=label_loss_train, inputs=last_layer_weight.parameters(), retain_graph=True)
                    norms.append(torch.norm(torch.mul(model.dynamic_weights[i], W_grad[0]))) # 将梯度范数加入norm列表
                norms = torch.stack(norms)

                # 5. 计算平均范数
                if torch.cuda.is_available():
                    mean_norm = np.mean(norms.data.cpu().numpy())
                else:
                    mean_norm = np.mean(norms.data.numpy())

                # 6. 计算梯度范数损失  
                #    = 梯度范数 - 常数项
                constant_term = torch.tensor(mean_norm * (inverse_train_rate ** 0.01), requires_grad=False)
                if torch.cuda.is_available():
                    constant_term = constant_term.cuda()

                grad_norm_loss = torch.sum(torch.abs(norms - constant_term))

                # 7. 计算grad_norm_loss在动态任务权重上的梯度
                #    之后通过optimizer更新该动态权重
                model.dynamic_weights.grad = torch.autograd.grad(grad_norm_loss, model.dynamic_weights)[0]


            #-------------------------------------------------------------------------------
            
            if epoch%100 == 0:
                print("epoch:{:4} --- [ 总loss:{:10.2f}, 任务loss:{} ]".format(epoch, loss_weighted.item(), [round(item.item(), 2) for item in label_loss_train]))
            

            optimizer.step()
        
        # renormalize
        normalize_coeff = model.task_num / torch.sum(model.dynamic_weights.data, dim=0)
        model.dynamic_weights.data = model.dynamic_weights.data * normalize_coeff


            
        # We tell PyTorch to NOT use autograd...
        with torch.no_grad():
            # Uses loader to fetch one mini-batch for validation
            for x_val, y_val in val_loader:
                
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                
                model.eval()
                yhat_val = model(x_val)
                y_val_t1, y_val_t2 = y_val[:, 0], y_val[:, 1]
                y_val_t3, y_val_t4 = y_val[:, 2], y_val[:, 3]

                yhat_t1_val, yhat_t2_val = yhat_val[0], yhat_val[1]
                yhat_t3_val, yhat_t4_val = yhat_val[2], yhat_val[3]

                # 对每个tower分别计算val_r2
                r2_t1_val = r2_score(y_val_t1.detach().cpu().numpy(), yhat_t1_val.detach().cpu().numpy())
                r2_t2_val = r2_score(y_val_t2.detach().cpu().numpy(), yhat_t2_val.detach().cpu().numpy())
                r2_t3_val = r2_score(y_val_t3.detach().cpu().numpy(), yhat_t3_val.detach().cpu().numpy())
                r2_t4_val = r2_score(y_val_t4.detach().cpu().numpy(), yhat_t4_val.detach().cpu().numpy())
                val_r2 = r2_t1_val + r2_t2_val + r2_t3_val + r2_t4_val
                # print('Validation阶段 r2_t1:{}, r2_t2:{}, r2_t3:{},r2_t4:{}'.format(r2_t1_val, r2_t2_val, r2_t3_val, r2_t4_val))
        
    return r2_t1_val, r2_t2_val, r2_t3_val, r2_t4_val
