import pandas as pd
import json
import numpy as np
import os
from random import shuffle
import csv
from pandas import read_csv
import math
import time


class generator():
    def __init__(self):
        with open('..\loc.json') as f:
            self.locations = json.load(f)
            # self.locations=tuple(self.locations)
            # print(self.locations)
            # print(len(self.locations))
        f.close()

    def find_data(self, data):
        output_data = []
        for num, i in enumerate(data):
            if num == 0: continue
            result = []
            ii = i.split(',')
            if [round(float(ii[1]), 5), float(ii[3])] in self.locations:
                result.extend([float(ii[5]), float(ii[15]) * math.pow(10, 6)])
                # output_data.append([round(float(ii[1]), 5), float(ii[3])])
                output_data.append(result)
        # print(output_data)
        # print(len(output_data))
        return output_data

    def get_location(self, data):
        minlen = 100000
        for i in data:
            # print(i)
            i = i.split(',')
            print(type(i))
            xx, zz = float(i[1]), float(i[3])
            if math.pow(xx - 0.105, 2) + math.pow(zz - 0.01, 2) < minlen:
                minlen = math.pow(xx - 0.105, 2) + math.pow(zz - 0.01, 2)
                x_min, z_min = xx, zz
        print(x_min, z_min)

    def write2cs(self, data, save_path):
        with open(save_path + 'all_information_80_5_0.csv', 'w', encoding='utf-8', newline='') as f:
            csv.writer(f).writerow(['force', 'strain'])
            for item in data:
                for j in item:
                    csv.writer(f).writerow(j)
        f.close()

    def get_data(self, data_path):
        csv_data_all = []
        num = 0
        for file in os.listdir(data_path):
            tic = time.time()
            with open(data_path + '/' + file) as f:
                num += 1
                data = f.readlines()
                data1 = self.find_data(data)  ## data1中保存每个csv中的90个点坐标
                # print(data1)
                # break
                csv_data_all.append(data1)  ## csv_data_all
            f.close()
            print("num=%d time=%6f" % (num, time.time() - tic))
        self.write2cs(csv_data_all, save_path)
        return csv_data_all
        # print(self.ids)
        # print(len(self.ids))

    def eulidean_metric(self, data):
        output = []
        for item in data:
            metric = 0
            for j in range(0, 3):
                metric += math.pow(float(item[j]) - float(item[j + 3]), 2)
            output.append([math.sqrt(metric), item[6], item[7]])
        return output


if __name__ == "__main__":
    data_path = r'F:\test_csv'
    save_path = r'F:\杂物\5600\5600_pytorch - 副本\output_data/'
    dataset = generator()
    csv_data_all = dataset.get_data(data_path)
    # print(len(csv_data_all))
