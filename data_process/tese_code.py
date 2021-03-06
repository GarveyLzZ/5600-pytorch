import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import torch.utils.data as Data

# 数据的导入
with open(r'F:\杂物\5600\output_data/lenth.csv', 'r') as f:
    forcedata = f.readlines()
# del forcedata[0]
# print(forcedata[0])
f.close()
X_data = []
y_data = []
channel_1 = []
channel_2 = []
for num, item in enumerate(forcedata):
    if num == 0:
        continue
    data = item.split(',')
    force = data[-2].replace('\n', '')
    del data[-2]
    channel_1.append(data[0])
    channel_2.append(data[1].strip('\n'))
    if len(channel_1) == 90:
        channel_1.extend(channel_2)
        X_data.extend([channel_1])
        y_data.append(force)
        channel_1 = []
        channel_2 = []
f.close()

X_train = []
X_test = []
y_train = []
y_test = []
shuffle = np.random.permutation(len(X_data))
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
y_train = np.array(y_train)
y_test = np.array(y_test).astype(np.float)

# 数据的标准化
scale = StandardScaler()
X_train_s = scale.fit_transform(X_train)
X_test_s = scale.transform(X_test)

print(type(X_train_s))

X_data.clear()
train_x = np.stack(((X_train_s[j].reshape(2, 1, -1) for j in range(0, len(X_train_s)))), axis=0)
test_x = np.stack(((X_test_s[j].reshape(2, 1, -1) for j in range(0, len(X_test_s)))), axis=0)
# train_xt = torch.from_numpy(X_train.astype(np.float32))

train_xt = torch.from_numpy(train_x.astype(np.float32))
train_yt = torch.from_numpy(y_train.astype(np.float32))
test_xt = torch.from_numpy(test_x.astype(np.float32))
# print(train_xt.shape)
# test_yt = torch.from_numpy(y_test.astype(np.float32))

train_data = Data.TensorDataset(train_xt, train_yt)
# test_data = Data.TensorDataset(test_xt, test_yt)
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=8,
    shuffle=True,
    drop_last=False
)
