from mimetypes import init
from model import AOE
from train import train
from read_file import read_file
import torch

# SEEDS = [1, 7, 15, 36, 21, 2, 32, 42, 25, 18]
SEEDS = [1, 7, 15, 32, 18]

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    '''
    进行的是多折验证
    如果只想单次,把SEEDS集合改一下就好
    '''
    # 读取数据
    X, y = read_file(filepath='./4tasks-encode.xlsx', iScaler=False)

    r2_t1_sum, r2_t2_sum, r2_t3_sum, r2_t4_sum = 0, 0, 0, 0
    for index, seed in enumerate(SEEDS):
        model = AOE(input_dim=18, hidden_dim=32, output_dim=1, task_num=4, expert_num=6)
        model = model.to(device)
        print("第{}个随机种子为{}".format(index, seed))

        r2_t1, r2_t2, r2_t3, r2_t4 = train(lr=1e-2, n_epochs=1500, seed=seed, model=model, X=X, y=y)

        print("当前随机种子下, r2为:{}".format([r2_t1, r2_t2, r2_t3, r2_t4]))
        r2_t1_sum = r2_t1_sum + r2_t1
        r2_t2_sum = r2_t2_sum + r2_t2
        r2_t3_sum = r2_t3_sum + r2_t3
        r2_t4_sum = r2_t4_sum + r2_t4

    print("五折验证的平均值为：{}".format([r2_t1_sum / 5, r2_t2_sum / 5, r2_t3_sum / 5, r2_t4_sum / 5]))


main()
