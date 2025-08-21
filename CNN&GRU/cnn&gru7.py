# python3.9
# time:2024/11/11
# CNN:Swiss注意力机制  GRU:dotproduct注意力机制

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torch.nn.utils import weight_norm
from torch.autograd import Variable
import math
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def onehot(x):
    # 将0变为0向量，其余变为对应的onehot编码
    # x:(batch_size, 50) -> (batch_size, 1, 50, 29)
    encode = torch.zeros((x.shape[0], 1, x.shape[1], 29))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] == 0:
                break
            else:
                encode[i, 0, j, x[i, j]-1] = 1
    encode = encode.float().cuda()

    return encode

# np.random.seed(60)
epsilon = 1e-8
batch_size = 100
batchsize_test = 500
length = 50
x_dimens = 20
data_false = pd.read_excel(r'data\data_false.xlsx')
data_true = pd.read_excel(r'data\data_true.xlsx')
data_false = data_false.values[:, 1:]
data_true = data_true.values[:, 1:]
templatevec = pd.read_excel(r'..\semantic\PCA20_idf.xlsx')
# templatevec = pd.read_excel(r'..\semantic\mean768.xlsx')
templatevec = templatevec.values
# 对模板向量进行归一化
# templatevec = (templatevec - np.min(templatevec, axis=1).reshape([-1, 1])) / (np.max(templatevec, axis=1) - np.min(templatevec, axis=1)).reshape([-1, 1])

x_false = np.zeros((data_false.shape[0], length))
y_false = data_false[:, 0]
x_true = np.zeros((data_true.shape[0], length))
y_true = data_true[:, 0]

# 将excel中的字符串转换为数字列表，同时将序列长度统一为50
for i in range(data_false.shape[0]):
    data_false[i, 1] = data_false[i, 1].split(",")
    data_false[i, 1] = [int(x) for x in data_false[i, 1]]
    if len(data_false[i, 1]) > length:
        data_false[i, 1] = data_false[i, 1][:length]
    else:
        data_false[i, 1] = data_false[i, 1] + [0 for x in range(length - len(data_false[i, 1]))]
    x_false[i, :] = data_false[i, 1]
for i in range(data_true.shape[0]):
    data_true[i, 1] = data_true[i, 1].split(",")
    data_true[i, 1] = [int(x) for x in data_true[i, 1]]
    if len(data_true[i, 1]) > length:
        data_true[i, 1] = data_true[i, 1][:length]
    else:
        data_true[i, 1] = data_true[i, 1] + [0 for x in range(length - len(data_true[i, 1]))]
    x_true[i, :] = data_true[i, 1]

# 将日志键序列转换为向量序列
sequence_false = np.zeros((x_false.shape[0], x_false.shape[1], x_dimens))
sequence_true = np.zeros((x_true.shape[0], x_true.shape[1], x_dimens))
for i in range(x_false.shape[0]):
    for j in range(x_false.shape[1]):
        if x_false[i, j] != 0:
            sequence_false[i, j, :] = templatevec[int(x_false[i, j]-1), :]
for i in range(x_true.shape[0]):
    for j in range(x_true.shape[1]):
        if x_true[i, j] != 0:
            sequence_true[i, j, :] = templatevec[int(x_true[i, j]-1), :]

# 构建训练集和测试集
x_false_train, x_false_test, y_false_train, y_false_test = train_test_split(sequence_false, y_false, test_size=0.9, random_state=30)#, random_state=30)  # 0.9，0.65 30：0.986 100：0.986 测试到320
x_true_train, x_true_test, y_true_train, y_true_test = train_test_split(sequence_true, y_true, test_size=0.99, random_state=70)#, random_state=70)  # 30 70
# print(x_true_train.shape,"111", x_false_train.shape)
# print(x_true_test.shape,"111", x_false_test.shape)

x_train = np.concatenate([x_false_train, x_true_train], axis=0).astype(dtype="int64")
x_test = np.concatenate([x_false_test, x_true_test], axis=0).astype(dtype="int64")
y_train = np.concatenate([y_false_train, y_true_train], axis=0).astype(dtype="int64")
y_test = np.concatenate([y_false_test, y_true_test], axis=0).astype(dtype="int64")

x_train = torch.tensor(x_train).cuda()
x_test = torch.tensor(x_test).cuda()
y_train = torch.tensor(y_train).cuda()
y_train = F.one_hot(y_train, 2).to(torch.float)  # 将标签变为onehot，异常为[0, 1]
y_test = torch.tensor(y_test).cuda()
y_test = F.one_hot(y_test, 2).to(torch.float)

torch_train = Data.TensorDataset(x_train, y_train)
torch_test = Data.TensorDataset(x_test, y_test)
train_loader = Data.DataLoader(
    dataset=torch_train,  # 数据集
    batch_size=batch_size,  # 批大小
    shuffle=True,  # 是否洗牌数据
    drop_last=True  # 是否丢掉最后一个不完整的批次
)
test_loader = Data.DataLoader(
    dataset=torch_test,  # 数据集
    batch_size=batchsize_test,  # 批大小
    shuffle=True,  # 是否洗牌数据
    drop_last=True  # 是否丢掉最后一个不完整的批次
)
# 调试数据
# x_false_train = torch.tensor(x_false_train).to(torch.int64).cuda()
# 搭建模型
channel_size = 16
h_gru = 6
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        # self.embed = nn.Embedding(30, 128, 0)
        self.wordembed = nn.Linear(x_dimens, channel_size)
        self.attention = nn.Linear(channel_size, channel_size)
        self.conv1 = weight_norm(nn.Conv1d(channel_size, channel_size, 1, stride=1, padding=0))
        self.conv2 = weight_norm(nn.Conv1d(channel_size, channel_size, 3, stride=1, padding=1))
        self.conv3 = weight_norm(nn.Conv1d(channel_size, channel_size, 5, stride=1, padding=2))
        self.fc1 = nn.Linear(3*channel_size, 2)

        self.gru = nn.GRU(channel_size, h_gru, dropout=0, batch_first=True)
        # self.gru = nn.LSTM(channel_size, h_gru, dropout=0, batch_first=True)
        self.fc2 = nn.Linear(h_gru, 2)
        self.fc3 = nn.Linear(3*channel_size + h_gru, 2)
        self.w_omega = Variable(
            torch.zeros(h_gru, h_gru))
        self.u_omega = Variable(torch.zeros(h_gru))
        self.dotline1 = nn.Linear(h_gru, h_gru)

    def forward(self, x, batchsize=batch_size):
        # x = F.one_hot(x, 30).float()  # (batch_size,1,50,30)
        # x = onehot(x)
        x = self.wordembed(x.float())  # (batch_size,1,50,128)
        # x = self.embed(x)  # (batch_size, 50, 128)
        x = x.reshape([batchsize, channel_size, length])
        x1 = self.conv1(x)  # (batch_size,128,50)
        x1 = F.leaky_relu(x1, negative_slope=0.1)
        x2 = self.conv2(x)
        x2 = F.leaky_relu(x2, negative_slope=0.1)
        x3 = self.conv3(x)
        x3 = F.leaky_relu(x3, negative_slope=0.1)
        x4, _ = self.gru(x.reshape([batchsize, length, channel_size]))  # (batch_size, length, h_gru)
        x4 = self.dotproduct1(x4, h_gru, batchsize)
        x_gru = torch.sum(x4, dim=2)
        x_cat = torch.cat([x1, x2, x3], dim=1)  # (batch_size,384,50)
        # x_cat = torch.cat([x1, x2, x3, x4.reshape([batchsize, h_gru, length])], dim=1)  # (batch_size,384,50)

        # 构造注意力权重
        for i in range(x.shape[2]):
            if i == 0:
                attn = F.tanh(self.attention(x[:, :, i].reshape([x.shape[0], 1, channel_size])))
            else:
                attn = torch.cat([attn, F.tanh(self.attention(x[:, :, i].reshape([x.shape[0], 1, channel_size])))], dim=1)  # (batch_size, 50, 128)
        # attn = F.softmax(attn, dim=1)
        attn1 = torch.cat([attn, attn, attn], dim=2).reshape([batchsize, 3*channel_size, length])
        x_cnn = torch.sum(x_cat * attn1, dim=2)  # (batch_size, 3*channel_size)

        # x_gru = self.attention1(x4, length)  # (batch_size, h_gru)
        # print(x_cnn.shape, "111", x_gru.shape)
        x = torch.cat([x_cnn, x_gru], dim=1)

        x = F.dropout(x, 0.5)
        x = x.reshape(batchsize, 3*channel_size+h_gru)
        x = self.fc3(x)
        # x = F.softmax(x, dim=1)
        return x

    def attention1(self, gru_output, seq_len):
        output_reshape = torch.Tensor.reshape(gru_output,
                                              [-1, h_gru])

        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega.to(device)))
        # attn_tanh = torch.mm(output_reshape, self.w_omega.to(device))
        attn_hidden_layer = torch.mm(
            attn_tanh, torch.Tensor.reshape(self.u_omega.to(device), [-1, 1]))
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer),
                                    [-1, seq_len])
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        alphas_reshape = torch.Tensor.reshape(alphas,
                                              [-1, seq_len, 1])
        state = gru_output
        attn_output = torch.sum(state * alphas_reshape, 1)
        return attn_output

    def dotproduct1(self, x, h, batch):
        y = torch.matmul(  # (batch_size, length, h_gru)
            F.softmax(torch.matmul(x, x.reshape([batch, h, length])) / h, dim=1),
            x.reshape([batch, length, h]))
        # x = x.reshape([batch, length, h])
        # y = torch.matmul(
        #     F.softmax(torch.matmul(self.dotline1(x), self.dotline1(x).reshape([batch, h, length])) / math.sqrt(h), dim=1),
        #     self.dotline1(x))
        y = y.reshape([batch, h, length])
        return y


# 创建网络模型
network = NN()
network = network.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 0.001
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
optimizer1 = torch.optim.Adam(network.parameters(), lr=0.0001)

# 记录训练用的参数
# 训练次数
total_train_step = 0
# 测试的次数
total_test_step = 0
# 训练的轮数
epoch = 200
# 指标图
accuracy_graph = torch.zeros((1, epoch))
P_graph = torch.zeros((1, epoch))
recall_graph = torch.zeros((1, epoch))
F1_graph = torch.zeros((1, epoch))
start = time.time()
for i in range(epoch):
    print("------第 {} 轮训练开始------".format(i))

    for data in train_loader:
        x, y = data
        # print(x.shape)
        outputs = network(x)
        # outputs = F.softmax(outputs, dim=1)
        loss = loss_fn(outputs, y)
        outputs = outputs.argmax(1)
        # print(y.shape)
        y = y.argmax(1)
        # print(outputs.argmax(1).shape)
        # print(y.shape)
        accuracy_train = (outputs == y).sum() / y.shape[0]
        TP = (outputs * y).sum()
        FP = (outputs * (1-y)).sum()
        FN = ((1 - outputs) * y).sum()
        P = (TP / (TP + FP)).item()
        recall = (TP / (TP + FN)).item()

        F1 = (2*P*recall / (P + recall + epsilon))
        # F1 = 0

        # 优化器优化模型
        if F1 < 1:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()

        total_train_step += 1
        if total_train_step % 50 == 0:
            print("训练次数: {}".format(total_train_step))
            print("TP:{}, FP:{}, FN:{}".format(TP.item(), FP.item(), FN.item()))
            print("TrainLoss: {}, Accuracy: {}, P: {}, Recall: {} F1: {}".format(loss.item(), accuracy_train, P, recall, F1))

    # 测试集
    total_test_loss = 0
    total_test_step = 0
    total_accuracy_test = 0
    total_P = 0
    total_recall = 0
    total_F1 = 0
    with torch.no_grad():
        for data in test_loader:
            x, y = data
            outputs = network(x, batchsize_test)
            # outputs = F.softmax(outputs, dim=1)
            loss = loss_fn(outputs, y)
            outputs = outputs.argmax(1)
            y = y.argmax(1)

            total_test_loss = total_test_loss + loss
            total_test_step += 1
            # print(total_test_step)
            accuracy_test = (outputs == y).sum() / y.shape[0]
            TP = (outputs * y).sum()
            FP = (outputs * (1 - y)).sum()
            FN = ((1 - outputs) * y).sum()
            P = (TP / (TP + FP)).item()
            recall = (TP / (TP + FN)).item()
            F1 = (2 * P * recall / (P + recall + epsilon))
            # F1 = 0
            total_accuracy_test = total_accuracy_test + accuracy_test
            total_P = total_P + P
            total_recall = total_recall + recall
            total_F1 = total_F1 + F1
            accuracy_graph[0, i] = (total_accuracy_test / total_test_step)
            P_graph[0, i] = (total_P/total_test_step)
            recall_graph[0, i] = (total_recall/total_test_step)
            F1_graph[0, i] = (total_F1/total_test_step)
        # if i % 100 == 0:
    print("TestLoss: {}, Accuracy: {}, P: {}, Recall: {} F1: {}".format(total_test_loss.item()/total_test_step, total_accuracy_test/total_test_step, total_P/total_test_step, total_recall/total_test_step, total_F1/total_test_step))
end = time.time()
print("运行时间为：{}min".format((end-start)/60))
# print(torch.squeeze(accuracy_graph))
plt.plot(range(epoch), torch.squeeze(accuracy_graph), label="Accuracy")
plt.plot(range(epoch), torch.squeeze(P_graph), label="P")
plt.plot(range(epoch), torch.squeeze(recall_graph), label="recall")
plt.plot(range(epoch), torch.squeeze(F1_graph), label="F1")
plt.ylim((0.8, 1))
plt.legend()
plt.show()

# torch.save(network.state_dict(), "without20_idf_b100.pth")
