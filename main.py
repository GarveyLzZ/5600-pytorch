import torch
from model.net import MLPregression, LeNet5, Lenet5_1X180, VGG16, conv1d
from model.ResNet import ResNet50
from sklearn.preprocessing import StandardScaler
import torch.utils.data as Data
# from data_process.data_process import data_process_fc, data_process_conv_2X1X90, data_process_conv_1X90, \
#     data_process_conv_2X1X90_test
from sklearn.metrics import mean_absolute_error
import math
import os
import csv
import numpy as np
from torch.optim import SGD, Adam
import torch.nn as nn
import matplotlib.pyplot as plt


def data_process():
    strain = []
    f = open("计算应变值_180.csv", 'r')
    strain_data = f.readlines()

    for i in strain_data:
        i = i.strip('\n')
        per_strain = i.split(',')
        print(len(per_strain))
        strain.append(per_strain)
    print(strain)
    strain = np.array(strain).astype(np.float)
    # scale = StandardScaler()
    # strain_train_s = scale.fit_transform(strain)
    f.close()

    f = open("计算力值_180.csv", 'r')
    force_data = f.readlines()
    force_data = force_data[0].strip('\n')
    force = force_data.split(',')
    force = np.array(force).astype(np.float)
    print(force)
    force_t = force / 90

    strain_t = torch.from_numpy(strain.astype(np.float32))
    force_t = torch.from_numpy(force_t.astype(np.float32))

    print(strain_t.shape,force_t.shape)

    train_data = Data.TensorDataset(strain_t, force_t)
    # test_data = Data.TensorDataset(test_xt, test_yt)
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=1,
        shuffle=True
    )
    return train_loader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
num = 0

if torch.cuda.is_available():
    model = MLPregression()
    model = model.to(device)
else:
    model = MLPregression()
    model = model.to(device)

# 参数初始化
learning_rate = 0.0001
train_epoch = 100
loss_fnc = nn.MSELoss()
train_rate = 0.9
weight_decay = 1e-5

train_loader = data_process()

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

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
        if step % 1 == 0:
            print("epoch:%d/%d:" % (epoch, train_epoch))
            print("     %d/%d\t" % (step // 2, int(len(train_loader) / 2)), end='')
            print("loss=%5f" % loss.item())
        train_num += b_x.size(0)
    # 平均损失函数
    train_loss_all.append(train_loss / train_num)
    print('-------------------------------------------------------------', train_loss_all[epoch])

    # plt.show()

x = [1250, 788, 972, 437, 510]
f = open('./output_data/all_information_80_5_0.csv', 'r')
force_data = f.readlines()
f.close()
test_strain = []
test_force = []
for i in x:
    end = i * 90
    begin = end - 89
    strain1 = []
    for j in range(begin, end + 1):
        force_strain = force_data[j].strip('\n')
        force_strain = force_strain.split(',')
        force, strain = float(force_strain[0]), float(force_strain[1])
        strain1.append(strain)

    test_strain.append(strain1)

    test_force.append(force)
for i in test_strain: print(i)
# print(test_strain)

# scale = StandardScaler()
# test_strain_s = scale.fit_transform(test_strain)

test_force = np.array(test_force).astype(np.float)

test_strain = np.array(test_strain)
test_strain = torch.from_numpy(test_strain.astype(np.float32))
test_force = test_force / 90
print(test_strain)
print(test_force * 90)

# 对测试集进行预测
# test_xt = test_xt.unsqueeze(1).cuda()
test_strain = test_strain.cuda()
# test_xt = test_xt[:, np.newaxis, :]
pre_y = model(test_strain).cuda()
pre_y = pre_y.data.cpu().numpy()
pre_y = pre_y.astype(np.float)
print(pre_y * 90)
total_force = 0
true_force = []
relative_error = []

for _ in range(0, len(pre_y)):
    total_force += (abs(pre_y[_] - test_force[_]) / test_force[_])
    true_force.append(test_force[_])
    relative_error.append(abs(pre_y[_] - test_force[_]) / test_force[_])
print(total_force, len(pre_y))
mae = mean_absolute_error(test_force, pre_y)
print("在测试集上的误差为：", mae * 90)
print("平均误差百分比为：%.3f%%" % ((total_force / len(pre_y)) * 100))

torch.save(model, 'logs/model_' + str((total_force / len(pre_y)) * 100) + '.pkl')

plt.figure(figsize=(12, 5))
plt.scatter(true_force, relative_error, c="r", label="Original Y")
plt.legend(loc="upper left")
plt.grid()
plt.xlabel("Force")
plt.ylabel("Relative Error")
plt.savefig("output_image/Scatter_" + str((total_force / len(pre_y)) * 100) + ".png")

# 可视化在测试集上真实值和预测值的差异
index = np.argsort(test_force)  # 元素从小到大排序后，提取排序后的下标
plt.figure(figsize=(12, 5))
plt.plot(np.arange(len(test_force)), test_force[index] * 90, "r", label="Original Y")
plt.scatter(np.arange(len(pre_y)), pre_y[index] * 90, s=3, c="b", label="Prediction")
plt.legend(loc="upper left")
plt.grid()
plt.xlabel("Index")
plt.ylabel("Y")
plt.savefig("output_image/Result_" + str((total_force / len(pre_y)) * 100) + ".png")
# plt.show()
num += 1
