# MTL-code
多任务学习相关代码
### 1 数据集
高熵合金，四个回归任务
### 2 算法思想
采用attention机制替代MMOE中的门网络，使得专家网络的权重分配可以达到自适应匹配
#### 2.1 具体步骤
TaskNet--> query

Expert --> key

alpha = query * key (矩阵相似度计算)

attention = alpha * value; value = key

TowerInput = query + atention

#### 2.2 GradNorm

![GradNorm算法思想](D:\WorkSpace\MTL-code\img\GradNorm算法思想.jpg)

### 3 文件解释

#### 3.1 AOE

该文件里的train函数未集成GradNorm

#### 3.2 AOE with GradNorm

train函数已经集成GradNorm

#### 3.3 img

存储一些相关笔记的图片

