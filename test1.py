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

ground_truth = [46.3707468, 34.9403094, 53.7043365, 40.7783745, 48.955869, 73.6725177, 71.6541291, 33.7903434,
                71.9324829, 50.6832318, 61.6284549, 36.7729839, 76.7833785, 53.5610502, 43.6965525, 27.8993826,
                42.0725583, 41.5172493, 24.6695526, 86.7510972]

ground_truth = [46.3707468,
                0.4438476,
                53.7043365, 40.7783745, 48.955869, 73.6725177, 71.6541291, 33.7903434,
                5.8591359,
                50.6832318, 61.6284549, 36.7729839, 76.7833785, 53.5610502, 43.6965525, 27.8993826,
                42.0725583, 41.5172493, 24.6695526, 86.7510972]

prediction = [46.30250752, 34.95365739, 53.63953471, 40.7616511, 48.92577767, 73.6573112, 71.63526893, 33.8070935,
              71.92253351, 50.63277304, 61.58773005, 36.78931296, 76.80417001, 53.49338651, 43.66345257, 27.89542168,
              42.06055373, 41.50478661, 24.67442662, 86.9459188]

'''
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.pool1(x))

        x = torch.relu(self.conv2(x))
        x = torch.relu(self.pool1(x))

        x = torch.relu(self.conv3(x))
        x = torch.relu(self.pool1(x))

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
'''
prediction = [46.37085557, 0.44513404, 53.7043798, 40.77923834, 48.95595789, 73.67260516, 71.6542536, 33.79062742,
              5.85998029, 50.68331122, 61.62871957, 36.77306414, 76.7836082, 53.56112838, 43.69661808, 27.89973199,
              42.07268, 41.51723206, 24.66995806, 86.75123334]
prediction = [46.36204183, 0.39044917, 53.69454145, 40.77003032, 48.94674718, 73.6602509, 71.64215684, 33.77969742,
              5.87312579, 50.67404687, 61.61798, 36.76512748, 76.77081943, 53.55129004, 43.68792504, 27.89633095,
              42.06416398, 41.50879115, 24.65956986, 86.73422813, ]
'''一个残差块，四个卷积层        x = torch.relu(self.conv1(x))
        x = torch.relu(self.pool1(x))

        x = torch.relu(self.conv2(x))
        x = torch.relu(self.pool1(x))

        x = torch.relu(self.conv3(x))
        x = torch.relu(self.pool1(x))

        x = torch.relu(self.conv4(x))'''
prediction=[46.35841548,  0.42954639,
 53.69320035,
 40.76589704,
 48.94397914,
 73.66462827,
 71.64586365,
 33.77765089,
  5.84524423,
 50.67173481,
 61.61868811,
 36.76049262,
 76.77591562,
 53.54989529,
 43.68445426,
 27.88640946,
 42.06033647,
 41.50476247,
 24.65645581,
 86.74399137,]


total_error = 0
for i in range(len(prediction)):
    error = abs(ground_truth[i] - prediction[i]) / ground_truth[i]
    total_error += error
    print("Error=%.5f%%" % (error * 100))
print("Average relatibe error=%.5f%%" % (total_error * 100 / len(prediction)))

'''
[[46.34393156] 
 [ 0.59194207]
 [53.55655789]
 [40.85054219]
 [48.93065393]
 [73.61635387]
 [71.54725492]
 [33.81864578]
 [ 6.01484567]
 [50.56380808]
 [61.53617263]
 [36.7915982 ]
 [76.90511227]
 [53.40881646]
 [43.69092107]
 [27.78336436]
 [42.1238324 ]
 [41.58145487]
 [24.54591125]
 [86.75857723]]
[0.38988062] 20
在测试集上的误差为： 0.08198298815906487
平均误差百分比为：1.949%

Process finished with exit code 0
'''
