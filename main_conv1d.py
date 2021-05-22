import torch
from model.net import MLPregression, LeNet5, Lenet5_1X180, VGG16, conv1d
from model.ResNet import ResNet50
from sklearn.preprocessing import StandardScaler
import torch.utils.data as Data
from data_process.data_process import data_process_fc, data_process_conv_1X90, \
    data_process_conv_2X1X90_test
from sklearn.metrics import mean_absolute_error
import math
import os
import csv
import numpy as np
from torch.optim import SGD, Adam
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
num = 0

if torch.cuda.is_available():
    model = conv1d()
    model = model.to(device)
    print(model)
else:
    model = conv1d()
    model = model.to(device)

# for i in model.modules():
#     print(i)


# 参数初始化
learning_rate = 0.0001
train_epoch = 300
loss_fnc = nn.MSELoss()
train_rate = 1
weight_decay = 1e-5

train_loader, test_xt, test_yt = data_process_conv_1X90(train_rate)

optimizer = Adam(model.parameters(), lr=learning_rate )

train_loss_all = []
print("Begin Training!")
for epoch in range(0, train_epoch):
    train_loss = 0
    train_num = 0
    for step, (b_x, b_y) in enumerate(train_loader):
        b_y = b_y.unsqueeze(1)
        # b_x = b_x[:, np.newaxis, :]
        b_x = b_x.to(device)
        output = model(b_x).to(device)
        b_y = b_y.cuda()
        loss = loss_fnc(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * b_x.size(0)
        # if step % 32 == 0:
        #     print("epoch:%d/%d:" % (epoch, train_epoch))
        #     print("     %d/%d\t" % (step // 32, int(len(train_loader) / 32)), end='')
        #     print("loss=%5f" % loss.item())
        train_num += b_x.size(0)

    test_strain = test_xt.cuda()
    # test_xt = test_xt[:, np.newaxis, :]
    pre_y = model(test_strain).cuda()
    pre_y = pre_y.data.cpu().numpy()
    pre_y = pre_y.astype(np.float)
    total_force = 0
    test_error = 0
    true_force = []
    relative_error = []

    for _ in range(0, len(pre_y)):
        total_force += abs(pre_y[_] - test_yt[_]) / test_yt[_]
        true_force.append(test_yt[_])
        test_error += abs(pre_y[_] - test_yt[_])
        relative_error.append(abs(pre_y[_] - test_yt[_]) / test_yt[_])
    mae = mean_absolute_error(test_yt, pre_y)
    print("平均误差百分比为：%.8f%%，测试集误差为：%.8f" % ((total_force / len(pre_y)) * 100, test_error))
    if (total_force / len(pre_y)) * 100 < 0.09:
        torch.save(model, 'logs1/model_' + str((total_force / len(pre_y)) * 100) + '.pkl')
        print(pre_y * 90)

    # 平均损失函数
    train_loss_all.append(train_loss / train_num)
    print("epoch:%d/%d:" % (epoch, train_epoch))
    print('----------------------------------------', train_loss_all[epoch])
    if train_loss / train_num < pow(10, -7): break

x = range(len(train_loss_all))
plt.plot(train_loss_all)
plt.show()

# 对测试集进行预测
# test_xt = test_xt.unsqueeze(1).cuda()
test_strain = test_xt.cuda()
# test_xt = test_xt[:, np.newaxis, :]
pre_y = model(test_strain).cuda()
pre_y = pre_y.data.cpu().numpy()
pre_y = pre_y.astype(np.float)
print(test_yt * 90)
print(pre_y * 90)
total_force = 0
true_force = []
relative_error = []

for _ in range(0, len(pre_y)):
    total_force += abs(pre_y[_] - test_yt[_]) / test_yt[_]
    true_force.append(test_yt[_])
    relative_error.append(abs(pre_y[_] - test_yt[_]) / test_yt[_])
print(total_force, len(pre_y))
mae = mean_absolute_error(test_yt, pre_y)
print("在测试集上的误差为：", mae * 90)
print("平均误差百分比为：%.8f%%" % ((total_force / len(pre_y)) * 100))

torch.save(model, 'logs/model_' + str((total_force / len(pre_y)) * 100) + '.pkl')

plt.figure(figsize=(12, 5))
plt.scatter(true_force, relative_error, c="r", label="Original Y")
plt.legend(loc="upper left")
plt.grid()
plt.xlabel("Force")
plt.ylabel("Relative Error")
plt.savefig("output_image/Scatter_" + str((total_force / len(pre_y)) * 100) + ".png")

# 可视化在测试集上真实值和预测值的差异
index = np.argsort(test_yt)  # 元素从小到大排序后，提取排序后的下标
plt.figure(figsize=(12, 5))
plt.plot(np.arange(len(test_yt)), test_yt[index] * 90, "r", label="Original Y")
plt.scatter(np.arange(len(pre_y)), pre_y[index] * 90, s=3, c="b", label="Prediction")
plt.legend(loc="upper left")
plt.grid()
plt.xlabel("Index")
plt.ylabel("Y")
plt.savefig("output_image/Result_" + str((total_force / len(pre_y)) * 100) + ".png")
# plt.show()
num += 1
