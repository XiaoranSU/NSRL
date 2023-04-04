from cache.DataLoader import DataLoader, DataLoaderZipf
from agents.DQNAgent_p4 import DQNAgent
import numpy as np
import copy
import matplotlib.pyplot as plt
import pandas as pd
from cache.DataLoader import DataLoaderPintos
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.utils.data as data


# import tensorflow.compat.v1 as tf

# tf.set_printoptions(profile="full")


# Auto-generated zipf simulation data
# dataloader = DataLoaderZipf(5000, 10000, 1.3, num_progs=5)
# dataloader1 = DataLoaderPintos(["d:/Code/learning/DRLCache/data/zipf1.csv"])
# dataloader2 = DataLoaderPintos(["d:/Code/learning/DRLCache/data/zipf2.csv"])

# Cache size: 5, 10, 50, ...
cache_size = 50
eprang = 1
num_presec = 100

sw = 1
agentnum = 2
# node_next = [[], [0], [1], [2], [1], [2], [1], [2]]
node_next = [[], []]
dataLoader = []
env = []
RL = []


step = 0
miss_rates = [[], [], [], [], [], [], [], [], [], []]
next_request = []
next_request_ = []
observation = []
iglist = []
modellist = []
shadowdata = []
# attackdata = []

trainpath1 = os.path.join('./data/agent'+str(1), 'agent.npy')
trainpath2 = os.path.join('./data/agent'+str(2), 'agent.npy')
testnpath1 = os.path.join('./data/agent'+str(3), 'agent.npy')
testnpath2 = os.path.join('./data/agent'+str(3), 'agent.npy')

path = [trainpath1, trainpath2]

testpath = [testnpath1, testnpath2]

truepath = os.path.join(
    './data/agent'+str(0), 'agent.npy')
model_path = os.path.join(
    './data/agent'+str(0), 'agent.pt')
fakepath = os.path.join(
    './data/agent'+str(4), 'agent.npy')

sd = [torch.tensor(np.load(trainpath1, allow_pickle=True)),
      torch.tensor(np.load(trainpath2, allow_pickle=True))]
# shadowdata = torch.tensor(np.load("2.npy", allow_pickle=True))
b = [torch.tensor(np.load(testnpath1, allow_pickle=True)),
     torch.tensor(np.load(testnpath2, allow_pickle=True))]
SM = []
a = []  # input output

for i in range(2):
    SM.append(DQNAgent(cache_size+1, cache_size*5+3,
                       learning_rate=0.01,
                       reward_decay=0,

                       # Epsilon greedy
                       e_greedy_min=(0.0, 0.1),
                       e_greedy_max=(0.2, 0.8),
                       e_greedy_init=(0.1, 0.5),
                       e_greedy_increment=(0.005, 0.01),
                       e_greedy_decrement=(0.005, 0.001),

                       history_size=50,
                       dynamic_e_greedy_iter=25,
                       reward_threshold=3,
                       explore_mentor='LRU',

                       replace_target_iter=100,
                       memory_size=10000,
                       batch_size=128,

                       output_graph=False,
                       #   privacy=True,
                       verbose=0
                       ))
    SM[i].load_memory(path[i])
    # print(env[0].n_features)
    for n in range(10000):
        # if (n % 200 == 0):
        #     print(n)
        SM[i].learn()
    ac = torch.ones(20000, 1)
    c = torch.cat(
        [sd[i][:, :cache_size*5+3], b[i][:, :cache_size*5+3]], dim=0)

    for n in range(20000):
        # print(c[n])
        ac[n] = SM[i].choose_action_t(c[n])

    d = torch.cat([c, ac], dim=1)
    a.append(d)


attackdata = torch.cat([a[0], a[1]], dim=0)
attackdata = attackdata.double()
# for i in range(2):
#     attackdata = torch.cat(
#         [attackdata, b[i]], dim=0)
y1_t = torch.ones(10000, 1)
y2_t = torch.zeros(10000, 1)
y_t = torch.cat((y1_t, y2_t, y1_t, y2_t), 0)
# print(y_t.size(dim=1))
train_labels = torch.tensor(y_t, dtype=torch.float32)


# 创建数据集和数据加载器
train_dataset = data.TensorDataset(attackdata, train_labels)
train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(254, 64).double()
        self.fc2 = nn.Linear(64, 32).double()
        self.fc3 = nn.Linear(32, 1).double()
        self.sigmoid = nn.Sigmoid()
        # self.float()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


net = Net()
print(net)
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 假设你有一个大小为 (n, 254) 的训练集 train_data 和一个大小为 (n,) 的标签集 train_labels
# 其中，train_labels 中的每个元素代表一个样本的类别，0 或 1

for epoch in range(10000):
    running_loss = 0.0
    # for i, data in enumerate(attackdata, 0):
    for i, d in enumerate(train_loader, 0):
        inputs, labels = d
        # inputs = data
        # labels = y_t[i]
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels.double())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    if (epoch % 100 == 0):
        print(f"Epoch {epoch+1}, loss: {running_loss/len(attackdata)}")

# net = nn.Sequential(
#     nn.Linear(cache_size*5+4, 64),  # 输入层与第一隐层结点数设置，全连接结构
#     torch.nn.Sigmoid(),  # 第一隐层激活函数采用sigmoid
#     nn.Linear(64, 32),  # 第一隐层与第二隐层结点数设置，全连接结构
#     torch.nn.Sigmoid(),  # 第一隐层激活函数采用sigmoid
#     nn.Linear(32, 2),  # 第二隐层与输出层层结点数设置，全连接结构
#     nn.Softmax(dim=1)  # 由于有两个概率输出，因此对其使用Softmax进行概率归一化
# )

# print(net)
# '''
# Sequential(
# (0): Linear(in_features=2, out_features=5, bias=True)
# (1): Sigmoid()
# (2): Linear(in_features=5, out_features=5, bias=True)
# (3): Sigmoid()
# (4): Linear(in_features=5, out_features=2, bias=True)
# (5): Softmax(dim=1)
# )'''

# # 配置损失函数和优化器
# optimizer = torch.optim.SGD(
#     net.parameters(), lr=0.01)  # 优化器使用随机梯度下降，传入网络参数和学习率
# loss_func = torch.nn.CrossEntropyLoss()  # 损失函数使用交叉熵损失函数

# # 模型训练
# num_epoch = 10000  # 最大迭代更新次数
# for epoch in range(num_epoch):
#     y_p = net(attackdata.float())  # 喂数据并前向传播

#     loss = loss_func(y_p, y_t.long())  # 计算损失
#     '''
#     PyTorch默认会对梯度进行累加，因此为了不使得之前计算的梯度影响到当前计算，需要手动清除梯度。
#     pyTorch这样子设置也有许多好处，但是由于个人能力，还没完全弄懂。
#     '''
#     optimizer.zero_grad()  # 清除梯度
#     loss.backward()  # 计算梯度，误差回传
#     optimizer.step()  # 根据计算的梯度，更新网络中的参数

#     if epoch % 1000 == 0:
#         print('epoch: {}, loss: {}'.format(
#             epoch, loss.data.item()))
#     # train shadow model
#     # 用收集的batchmemory训练影子模型
torch.save(net, 'net.pt')
# 训练完成，进行测试
Victim = DQNAgent(cache_size+1, cache_size*5+3,
                  learning_rate=0.01,
                  reward_decay=0,

                  # Epsilon greedy
                  e_greedy_min=(0.0, 0.1),
                  e_greedy_max=(0.2, 0.8),
                  e_greedy_init=(0.1, 0.5),
                  e_greedy_increment=(0.005, 0.01),
                  e_greedy_decrement=(0.005, 0.001),

                  history_size=50,
                  dynamic_e_greedy_iter=25,
                  reward_threshold=3,
                  explore_mentor='LRU',

                  replace_target_iter=100,
                  memory_size=10000,
                  batch_size=128,

                  output_graph=False,
                  #   privacy=True,
                  verbose=0
                  )
Victim.load_memory(truepath)
Victim.load_model(model_path)
true = torch.tensor(np.load(truepath, allow_pickle=True))
fake = torch.tensor(np.load(fakepath, allow_pickle=True))
ac = torch.ones(20000, 1)
c = torch.cat(
    [true[:, :cache_size*5+3], fake[:, :cache_size*5+3]], dim=0)

for n in range(20000):
    ac[n] = Victim.choose_action_t(c[n])

test_data = torch.cat([c, ac], dim=1)
test_data = test_data.double()
y3_t = torch.ones(10000, 1)
y4_t = torch.zeros(10000, 1)
test_labels = torch.cat((y3_t, y4_t), 0)

test_labels = torch.tensor(test_labels, dtype=torch.float32)
test_dataset = data.TensorDataset(test_data, test_labels)
test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# print(type(test_data))
# print(type(test_labels))
# print(test_data.shape)
# print(test_labels.shape)

# action_values = net(d.float())
# pred = action_values.argmax(dim=1, keepdim=True)
# correct = 0
# for n in range(10000):
#     if pred[n] == 1:
#         correct += 1
# for n in range(10000):
#     if pred[n+10000] == 0:
#         correct += 1
# # action = torch.argmax(action_values, axis=1).item()

# print(correct, correct/20000)
# torch.save(net, 'net.pkl')
# RL.plot_cost()
# 假设你有一个大小为 (m, 254) 的测试集 test_data 和一个大小为 (m,) 的标签集 test_labels
# 其中，test_labels 中的每个元素代表一个样本的类别，0 或 1

correct = 0
total = 0
with torch.no_grad():
    for i, d in enumerate(test_loader, 0):
        inputs, labels = d
        outputs = net(inputs)
        predicted = (outputs > 0.5).float()
        total += 1
        correct += (predicted == labels).sum().item()

accuracy = correct / 20000
print(correct, total)
print(f"Test accuracy: {accuracy}")
