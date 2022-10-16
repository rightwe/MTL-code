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
        c = 0
        # print("Epoch: {}".format(epoch))
        r2_t1_val, r2_t2_val, r2_t3_val, r2_t4_val = 0,0,0,0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device).to(torch.float32)
            y_batch = y_batch.to(device).to(torch.float32)
            yhat = model(x_batch)
            y_batch_t1, y_batch_t2 = y_batch[:, 0], y_batch[:, 1]
            y_batch_t3, y_batch_t4 = y_batch[:, 2], y_batch[:, 3]
            yhat_t1, yhat_t2 = yhat[0].to(torch.float32), yhat[1].to(torch.float32)
            yhat_t3, yhat_t4 = yhat[2].to(torch.float32), yhat[3].to(torch.float32)

            # 对每个tower分别计算train_r2
            r2_t1 = r2_score(y_batch_t1.detach().cpu().numpy(), yhat_t1.detach().cpu().numpy())
            r2_t2 = r2_score(y_batch_t2.detach().cpu().numpy(), yhat_t2.detach().cpu().numpy())
            r2_t3 = r2_score(y_batch_t3.detach().cpu().numpy(), yhat_t3.detach().cpu().numpy())
            r2_t4 = r2_score(y_batch_t4.detach().cpu().numpy(), yhat_t4.detach().cpu().numpy())
            train_r2 = r2_t1 + r2_t2 + r2_t3 + r2_t4
            # print('训练阶段 r2_t1:{}, r2_t2:{}, r2_t3:{},r2_t4:{}'.format(r2_t1,r2_t2,r2_t3,r2_t4))
            # 对每个tower分别计算损失，一共有4个
            loss_t1 = loss_fn(yhat_t1, y_batch_t1.view(-1, 1))
            loss_t2 = loss_fn(yhat_t2, y_batch_t2.view(-1, 1))
            loss_t3 = loss_fn(yhat_t3, y_batch_t3.view(-1, 1))
            loss_t4 = loss_fn(yhat_t4, y_batch_t4.view(-1, 1))
            # loss = 0.15*loss_t1 + 0.15*loss_t2 + 0.25*loss_t3 + 0.45*loss_t4
            loss = loss_t1 + loss_t2 + loss_t3 + loss_t4
            if epoch%100 == 0:
                print("epoch:{:4}----总loss:{:10.2f}, 任务loss:{:7.2f},{:7.2f},{:7.2f},{:7.2f}".format(epoch, loss, loss_t1, loss_t2, loss_t3, loss_t4))
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            
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
