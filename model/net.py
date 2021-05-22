import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.nn import BatchNorm1d


class MLPregression(nn.Module):
    def __init__(self):
        super(MLPregression, self).__init__()
        # 定义一个隐藏层
        self.fc1 = nn.Linear(in_features=90, out_features=150, bias=True)
        self.fc2 = nn.Linear(150, 300)
        self.fc3 = nn.Linear(300, 900)
        self.fc4 = nn.Linear(900, 300)
        # self.fc5 = nn.Linear(600, 300)
        self.predict = nn.Linear(300, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        # x = torch.sigmoid(self.fc5(x))
        output = self.predict(x)
        return output


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 输入图像为1×90×2 ==> (batch,2,1,90)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(1, 5), padding=0, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(6)
        # 经过卷积后变为1×86×6，经过池化后变为1×43×6 ==> (batch,6,1,43)
        self.conv2 = nn.Conv2d(6, 16, (1, 4), stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        # 经过卷积后为1×40×16，经过池化后变为1×20×16  ==>(16,1,20)
        self.conv3 = nn.Conv2d(16, 32, (1, 3), padding=(0, 1), stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        # 经过卷积后为1×18×16 ==>(16,1,18)
        self.conv4 = nn.Conv2d(32, 32, (1, 3), padding=(0, 1), stride=1)
        self.bn4 = nn.BatchNorm2d(32)

        # 转化为全连接为
        self.fc1 = nn.Linear(1312, 640)
        self.fc2 = nn.Linear(640, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x1):
        x = torch.sigmoid(self.conv1(x1))
        x = self.bn1(x)
        # x = nn.MaxPool2d((1, 2), 2)(x)

        x = torch.sigmoid(self.conv2(x))
        x = self.bn2(x)
        # x = nn.MaxPool2d((1, 2), 2)(x)
        #
        x = torch.sigmoid(self.conv3(x))
        x = self.bn3(x)

        x = torch.sigmoid(self.conv4(x))
        x = self.bn4(x)
        x = nn.MaxPool2d((1, 2), 2)(x)

        x = x.view(x.size(0), -1)

        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        output = self.fc3(x)

        # print("output=", output.size(0), output.size(1))
        # print('-------------------------------------------')

        return output


class Lenet5_1X180(nn.Module):
    def __init__(self):
        super(Lenet5_1X180, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(1, 5), stride=1)
        self.bn1 = nn.BatchNorm2d(6)
        # 经过卷积后变为1×176×6，经过池化后变为1×88×6 ==> (batch,6,1,88)
        self.conv2 = nn.Conv2d(6, 16, (1, 5), stride=1)
        self.bn2 = nn.BatchNorm2d(16)
        # 经过卷积后为1×88×16，经过池化后变为1×44×16  ==>(16,1,44)
        self.conv3 = nn.Conv2d(16, 32, (1, 3), stride=1)
        self.bn3 = nn.BatchNorm2d(32)
        # 经过卷积后为1×22×16 ==>(16,1,22)
        self.conv4 = nn.Conv2d(32, 64, (1, 3), stride=1)
        self.bn4 = nn.BatchNorm2d(64)

        # 转化为全连接为
        self.fc1 = nn.Linear(2816, 480)
        self.fc2 = nn.Linear(480, 120)
        self.fc3 = nn.Linear(120, 1)

    def forward(self, x):
        x = torch.sigmoid(self.conv1(x))
        x = self.bn1(x)
        x = nn.MaxPool2d((1, 2), 2)(x)

        x = torch.sigmoid(self.conv2(x))
        x = self.bn2(x)
        x = nn.MaxPool2d((1, 2), 2)(x)

        x = torch.sigmoid(self.conv3(x))
        x = self.bn3(x)

        x = torch.sigmoid(self.conv4(x))
        x = self.bn4(x)

        x = x.view(x.size(0), -1)

        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        output = self.fc3(x)
        return output


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=(1, 3), stride=1)
        self.conv1_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        # 经过卷积后变为1×176×6，经过池化后变为1×88×6 ==> (batch,6,1,88)
        self.conv2 = nn.Conv2d(64, 128, (1, 3), padding=(0, 1), stride=1)
        self.conv2_1 = nn.Conv2d(128, 128, (1, 3), padding=(0, 1), stride=1)
        self.bn2 = nn.BatchNorm2d(128)
        # 经过卷积后为1×88×16，经过池化后变为1×44×16  ==>(16,1,44)
        self.conv3 = nn.Conv2d(128, 256, (1, 3), padding=(0, 1), stride=1)
        self.conv3_1 = nn.Conv2d(256, 256, (1, 3), padding=(0, 1), stride=1)
        self.bn3 = nn.BatchNorm2d(256)
        # 经过卷积后为1×22×16 ==>(16,1,22)
        self.conv4 = nn.Conv2d(256, 128, (1, 3), padding=(0, 1), stride=1)
        self.conv4_1 = nn.Conv2d(512, 512, (1, 3), padding=(0, 1), stride=1)
        self.conv4_2 = nn.Conv2d(512, 512, (1, 3), padding=(0, 1), stride=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 64, (1, 3), padding=(0, 1), stride=1)
        self.conv5_1 = nn.Conv2d(512, 512, (1, 3), padding=(0, 1), stride=1)
        self.conv5_2 = nn.Conv2d(512, 512, (1, 3), padding=(0, 1), stride=1)
        self.bn5 = nn.BatchNorm2d(64)

        # 转化为全连接为
        self.fc1 = nn.Linear(2816, 480)
        self.fc2 = nn.Linear(480, 120)
        self.fc3 = nn.Linear(120, 1)

    def forward(self, x):
        x = torch.sigmoid(self.conv1(x))
        # x = torch.sigmoid(self.conv1_1(x))
        x = self.bn1(x)
        # x = nn.MaxPool2d((1, 2), 2)(x)

        x = torch.sigmoid(self.conv2(x))
        # x = torch.sigmoid(self.conv2_1(x))
        x = self.bn2(x)
        # x = nn.MaxPool2d((1, 2), 2)(x)

        x = torch.sigmoid(self.conv3(x))
        # x = torch.sigmoid(self.conv3_1(x))
        x = self.bn3(x)
        # x = nn.MaxPool2d((1, 2), 2)(x)

        x = torch.sigmoid(self.conv4(x))
        # x = torch.sigmoid(self.conv4_1(x))
        # x = torch.sigmoid(self.conv4_2(x))
        x = self.bn4(x)
        # x = nn.MaxPool2d((1, 2), 2)(x)

        x = torch.sigmoid(self.conv5(x))
        # x = torch.sigmoid(self.conv5_1(x))
        # x = torch.sigmoid(self.conv5_2(x))
        x = self.bn5(x)
        x = nn.MaxPool2d((1, 2), 2)(x)

        x = x.view(x.size(0), -1)

        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        output = self.fc3(x)
        return output


class conv1d(nn.Module):
    def __init__(self):
        super(conv1d, self).__init__()
        self.bottleNet = nn.Sequential(
            # nn.Conv1d(in_channels=1, out_channels=16, kernel_size=1, stride=1, bias=False),
            # nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, bias=False),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=8, out_channels=1, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(1),
        )

        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1)

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.bn1 = nn.BatchNorm1d(6)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm1d(16)
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1)

        self.dropout = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(192, 960)
        self.fc2 = nn.Linear(960, 240)
        self.fc3 = nn.Linear(240, 1)

        self.fc = nn.Linear(192, 1)

    def forward(self, x):
        x1 = self.bottleNet(x)
        x = self.conv(x)
        x = torch.relu(x1 + x)
        # # x = torch.relu(self.conv1(x))
        #
        # x2 = self.bottleNet(x)
        # x = self.conv(x)
        # x = torch.relu(x2 + x)
        # # #
        # # x = torch.relu(self.conv2(x))
        #
        # x3 = self.bottleNet(x)
        # x = self.conv(x)
        # x = torch.relu(x3 + x)
        #

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.pool1(x))

        x = torch.relu(self.conv2(x))
        x = torch.relu(self.pool1(x))

        x = torch.relu(self.conv3(x))
        x = torch.relu(self.pool1(x))

        x = torch.relu(self.conv4(x))

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = self.fc1(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # x = self.dropout(x)
        # x = self.fc3(x)
        return x
