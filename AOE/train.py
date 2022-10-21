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


def train(lr=1e-2, n_epochs=1000, loss_fn=loss_fn, seed=1, X=None, y=None, model=None):

    optimizer = optim.Adam(model.parameters(), lr=lr) # optimizer一定要放在循环里，因为优化器是和参数绑定的
    X_train, X_test, y_train, y_test = train_test_split(X.astype(np.float32), y, test_size=0.33, random_state=seed) 
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

            if epoch%100 == 0:
                print("epoch:{:4} --- [ 总loss:{:10.2f}, 任务loss:{} ]".format(epoch, loss_weighted.item(), [round(item.item(), 2) for item in label_loss_train]))
            

            optimizer.step()

            
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
