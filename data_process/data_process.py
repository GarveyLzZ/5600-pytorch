import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
import random


def regulate_tensor(data):
    train_xt = []
    numpy_data = np.ones((1, 90, 2))
    # print(data[0])
    for i, item in enumerate(data):
        # print(data[i].shape)
        channel1 = data[i][:90]
        channel2 = data[i][90:]
        # print(channel2.shape)
        channel1 = np.expand_dims(channel1, axis=0)
        channel2 = np.expand_dims(channel2, axis=0)
        # print(item)
        # print(channel1,'---',channel2)
        # print(channel1.shape)
        # print(numpy_data[:, :, 0].shape)
        numpy_data[:, :, 0] = channel1
        numpy_data[:, :, 1] = channel2
        numpy_new = np.transpose(numpy_data, (2, 0, 1))
        # print(numpy_data)
        # 2,1,90
        train_xt.append(numpy_new)
    # stack_data = np.stack((train_xt[j] for j in range(0, len(train_xt))), axis=0)
    # print(stack_data.shape)
    return train_xt


def data_process_fc(train_rate):
    # 数据的导入
    with open(r'F:\杂物\5600\5600_pytorch - 副本\output_data/all_information_80_5_0.csv', 'r') as f:
        forcedata = f.readlines()
    f.close()
    # del forcedata[0]
    # print(forcedata[0])
    strain = []
    X_data = []
    y_data = []
    for num, item in enumerate(forcedata):
        if num == 0: continue
        data = item.split(',')
        force = data[-2]
        strain.append(data[-1].strip('\n'))
        if len(strain) == 90:
            X_data.append(strain)
            y_data.append(force)
            strain = []

    X_train = []
    X_test = []
    y_train = []
    y_test = []
    with open('random.txt', 'r') as f:
        shuffle = f.readlines()
    shuffle = list(np.array(shuffle).astype(np.int))
    print(len(shuffle))
    train_num = int(len(shuffle) * train_rate)
    x = [800, 766, 980, 659, 863, 1414, 1356, 508, 1080, 915, 1136, 578, 1477, 977, 737, 379, 697, 680, 304, 1707]
    # x = [1406, 53, 215, 776, 632, 1143, 1430, 1796, 1185, 612, 316, 1246, 179, 183, 1379, 1269, 925, 629, 166, 826]
    shuffle = set(shuffle) - set(x)
    # for i in shuffle[int(len(shuffle) * (1-0.002)):]:
    for i in x:
        print(i)
        X_test.append(X_data[i - 1])
        y_test.append(y_data[i - 1])
    for i in shuffle:
        X_train.append(X_data[i])
        y_train.append(y_data[i])
    print("训练数据量：=%d" % (len(y_train)))
    print("测试数据量：=%d" % (len(y_test)))
    # 将训练数据转化为numpy格式
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train).astype(np.float)
    y_test = np.array(y_test).astype(np.float)

    # 数据的标准化
    scale = StandardScaler()
    X_train_s = scale.fit_transform(X_train)
    y_train_s = y_train / 90
    X_test_s = scale.transform(X_test)
    y_test_s = y_test / 90
    # train_xt = torch.from_numpy(X_train.astype(np.float32))

    train_xt = torch.from_numpy(X_train_s.astype(np.float32))
    train_yt = torch.from_numpy(y_train_s.astype(np.float32))
    test_xt = torch.from_numpy(X_test_s.astype(np.float32))
    # test_yt = torch.from_numpy(y_test.astype(np.float32))

    train_data = Data.TensorDataset(train_xt, train_yt)
    # test_data = Data.TensorDataset(test_xt, test_yt)
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=1,
        shuffle=True
    )
    print(y_test_s * 90)
    return train_loader, test_xt, y_test_s


def data_process_3X15x2(train_rate):
    # 数据的导入
    with open(r'F:\杂物\5600\5600_pytorch - 副本\output_data/all_information_多点.csv', 'r') as f:
        forcedata = f.readlines()
    f.close()

    # del forcedata[0]
    # print(forcedata[0])
    layer1 = np.zeros((3, 15))
    layer2 = np.zeros_like(layer1)
    for num, item in enumerate(forcedata):
        if num == 0: continue
        data = item.split(',')
        force = data[-2]
        strain.append(data[-1].strip('\n'))
        if len(strain) == 90:
            X_data.append(strain)
            y_data.append(force)
            strain = []

    X_train = []
    X_test = []
    y_train = []
    y_test = []
    with open('random1.txt', 'r') as f:
        shuffle = f.readlines()
    shuffle = np.array(shuffle).astype(np.int)
    train_num = int(len(shuffle) * train_rate)
    for i in shuffle[train_num:]:
        X_test.append(X_data[i])
        y_test.append(y_data[i])
    for i in shuffle[:train_num]:
        X_train.append(X_data[i])
        y_train.append(y_data[i])
    # 将训练数据转化为numpy格式
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train).astype(np.float)
    y_test = np.array(y_test).astype(np.float)

    # 数据的标准化
    scale = StandardScaler()
    X_train_s = scale.fit_transform(X_train)
    y_train_s = y_train / 90
    X_test_s = scale.transform(X_test)
    y_test_s = y_test / 90
    # train_xt = torch.from_numpy(X_train.astype(np.float32))

    train_xt = torch.from_numpy(X_train_s.astype(np.float32))
    train_yt = torch.from_numpy(y_train_s.astype(np.float32))
    test_xt = torch.from_numpy(X_test_s.astype(np.float32))
    # test_yt = torch.from_numpy(y_test.astype(np.float32))

    train_data = Data.TensorDataset(train_xt, train_yt)
    # test_data = Data.TensorDataset(test_xt, test_yt)
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=1,
        shuffle=True
    )
    return train_loader, test_xt, y_test_s


def data_process_conv_1X90(train_rate):  # 一维卷积 #
    # 数据的导入
    with open(r'F:\杂物\5600\5600_pytorch - 副本\output_data/all_information_80_5_0.csv', 'r') as f:
        forcedata = f.readlines()
    f.close()
    # del forcedata[0]
    # print(forcedata[0])
    strain = []
    X_data = []
    y_data = []
    for num, item in enumerate(forcedata):
        if num == 0: continue
        data = item.split(',')
        force = data[-2]
        strain.append(data[-1].strip('\n'))
        if len(strain) == 90:
            X_data.append(strain)
            y_data.append(force)
            strain = []

    X_train = []
    X_test = []
    y_train = []
    y_test = []
    with open('random.txt', 'r') as f:
        shuffle = f.readlines()
    shuffle = list(np.array(shuffle).astype(np.int))
    # print(len(shuffle))
    train_num = int(len(shuffle) * train_rate)
    x = [800, 766, 980, 659, 863, 1414, 1356, 508, 1080, 915, 1136, 578, 1477, 977, 737, 379, 697, 680, 304, 1707]
    # x = [800, 535, 980, 659, 863, 1414, 1356, 508, 1363, 915, 1136, 578, 1477, 977, 737, 379, 697, 680, 304, 1707]
    # x = [1406, 53, 215, 776, 632, 1143, 1430, 1796, 1185, 612, 316, 1246, 179, 183, 1379, 1269, 925, 629, 166, 826]
    shuffle = set(shuffle) - set(x)
    # for i in shuffle[int(len(shuffle) * (1-0.002)):]:
    for i in x:
        # print(i)
        X_test.append(X_data[i - 1])
        y_test.append(y_data[i - 1])
    for i in shuffle:
        X_train.append(X_data[i])
        y_train.append(y_data[i])
    print("训练数据量：=%d" % (len(y_train)))
    print("测试数据量：=%d" % (len(y_test)))
    # 将训练数据转化为numpy格式
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train).astype(np.float)
    y_test = np.array(y_test).astype(np.float)

    # 数据的标准化
    scale = StandardScaler()
    X_train_s = scale.fit_transform(X_train)
    y_train_s = y_train / 90
    X_test_s = scale.transform(X_test)
    y_test_s = y_test / 90
    # train_xt = torch.from_numpy(X_train.astype(np.float32))

    X_train_s = X_train_s[:, np.newaxis, :]
    X_test_s = X_test_s[:, np.newaxis, :]
    print(X_train_s.shape)

    train_xt = torch.from_numpy(X_train_s.astype(np.float32))
    train_yt = torch.from_numpy(y_train_s.astype(np.float32))
    test_xt = torch.from_numpy(X_test_s.astype(np.float32))
    # test_yt = torch.from_numpy(y_test.astype(np.float32))

    train_data = Data.TensorDataset(train_xt, train_yt)
    # test_data = Data.TensorDataset(test_xt, test_yt)
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=2,
        shuffle=True
    )
    print(y_test_s * 90)
    return train_loader, test_xt, y_test_s


def data_process_conv_2X1X90_test(train_rate):
    with open(r'F:\杂物\5600\output_data/lenth.csv', 'r') as f:
        forcedata = f.readlines()
    # del forcedata[0]
    # print(forcedata[0])
    f.close()
    X_data = []
    y_data = []
    channel_1 = []
    channel_2 = []
    channel1_data = []
    channel2_data = []
    for num, item in enumerate(forcedata):
        if num == 0:
            continue
        data = item.split(',')
        force = data[-2].replace('\n', '')
        del data[-2]
        channel_1.append(data[0])
        if data[1].strip('\n') == '0':
            channel_2.append(random.uniform(0.0001, 0.001) + 0.0001)
        else:
            channel_2.append(data[1].strip('\n'))
        if len(channel_1) == 90:
            channel1_data.append(channel_1)
            channel2_data.append(channel_2)
            y_data.append(force)
            channel_1 = []
            channel_2 = []
    f.close()
    # print(len(channel1_data))
    # print(channel1_data[0])
    # print(len(channel1_data[0]))
    # print(len(X_data))

    # In[48]:

    X_data.clear()
    scale = StandardScaler()
    lenth_s = scale.fit_transform(channel1_data)
    strain_s = scale.fit_transform(channel2_data)
    print(lenth_s.shape)
    print(strain_s.shape)
    # for i in range(0, len(lenth_s)):
    #     data1 = np.hstack((lenth_s[i], strain_s[i]))
    #     X_data.append(data1.reshape(1, 2, -1))
    # print(len(X_data))
    # print(X_data[0].shape)
    # print(type(X_data[0]))
    # print("---------------------------")

    X_train_channel1 = []
    X_test_channel1 = []
    X_train_channel2 = []
    X_test_channel2 = []
    y_train = []
    y_test = []

    with open('random.txt', 'r') as f:
        shuffle = f.readlines()
    f.close()
    # shuffle = np.array(shuffle)
    train_num = int(len(shuffle) * train_rate)

    for i in range(0, len(shuffle[train_num:])):
        num = int(shuffle[train_num:][i].strip('\n'))
        X_test_channel1.append(lenth_s[num])
        X_test_channel2.append(strain_s[num])
        y_test.append(y_data[num])

    for i in range(0, len(shuffle[:train_num])):
        num = int(shuffle[:train_num][i].strip('\n'))
        X_train_channel1.append(lenth_s[num])
        X_train_channel2.append(strain_s[num])
        y_train.append(y_data[num])
    # print(type(X_train))
    # print(len(X_train))
    # print(type(X_train[0]))

    # In[49]:

    train_x_1 = np.stack(((X_train_channel1[j].reshape(1, 1, -1) for j in range(0, len(X_train_channel1)))), axis=0)
    train_x_2 = np.stack(((X_train_channel2[j].reshape(1, 1, -1) for j in range(0, len(X_train_channel2)))), axis=0)

    test_x_1 = np.stack(((X_test_channel1[j].reshape(1, 1, -1) for j in range(0, len(X_test_channel1)))), axis=0)
    test_x_2 = np.stack(((X_test_channel2[j].reshape(1, 1, -1) for j in range(0, len(X_test_channel2)))), axis=0)

    # train_xt = torch.from_numpy(X_train.astype(np.float32))

    y_train = np.array(y_train)
    y_test = np.array(y_test).astype(np.float)

    train_xt1 = torch.from_numpy(train_x_1.astype(np.float32))
    train_xt2 = torch.from_numpy(train_x_2.astype(np.float32))
    print(train_xt1.shape, train_xt2.shape)
    train_yt = torch.from_numpy(y_train.astype(np.float32))
    test_xt1 = torch.from_numpy(test_x_1.astype(np.float32))
    test_xt2 = torch.from_numpy(test_x_2.astype(np.float32))
    # print(train_xt.shape)
    # test_yt = torch.from_numpy(y_test.astype(np.float32))

    train_data = Data.TensorDataset(train_xt1, train_xt2, train_yt)
    # test_data = Data.TensorDataset(test_xt, test_yt)
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=8,
        shuffle=True,
        drop_last=False
    )
    return train_loader, test_xt1, test_xt2, y_test

# data = np.ones((1, 90, 2))
# data_process_conv_1X90(0.9)
# data_process_conv_1X90(1)
