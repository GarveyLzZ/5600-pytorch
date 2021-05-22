import numpy as np
import csv
from numpy import *


def append_data2numpy(num1, xx, result_numpy):
    num0 = num1 - 90 + 1
    num1 += 1
    strain_list = []
    for i in range(num0, num1):
        item = force_data[i]
        item = item.replace('\n', '')
        data = item.split(',')  # 将str类型转化为一个list，每个list中有两个元素，分别为力和应变
        # print(data)
        strain_list.append(float(data[1]))
        v = float(data[0])
        # print(i, data[0])
    V.append(v)
    strain_list = np.array(strain_list)
    result_numpy[xx, :] = strain_list
    # print(result_numpy)
    # print(V)
    # print(strain_list)
    # print(num0, num)


# 读入数据，将数据放入list中，每个list中存储的为str类型的字符
f = open("output_data/all_information_80_5_0.csv", 'r')
force_data = f.readlines()
f.close()

force_list = list(range(0, 90))
len1 = 0

result_numpy = np.zeros((90, 90))

V = []

for num, item in enumerate(force_data):
    if num == 0: continue  # 第0行是表头，略过不看
    # print(num)
    # print(item)
    # print(type(item))
    item = item.replace('\n', '')
    data = item.split(',')  # 将str类型转化为一个list，每个list中有两个元素，分别为力和应变
    if num % 90 == 0 and num != 1:
        # print(num, data)
        if int(round(float(data[0]), 3)) in force_list:
            # print(data[0])
            # print(num)
            append_data2numpy(num, len1, result_numpy)
            len1 += 1
            force_list.remove(int(round(float(data[0]), 3)))
            # print(int(round(float(data[0]), 3)),force_list)
        # print(data)

print(result_numpy)
print(V)
# f = open('计算数据.csv', 'w', newline='')
# csv_writer = csv.writer(f)
# for i in range(1, 91):
#     x = result_numpy.T[i - 1:i]
#     x = np.squeeze(x)
#     x = x.tolist()
#     csv_writer.writerow(x)
# f.close()

# f = open('计算力值.csv', 'w', newline='')
# csv_writer = csv.writer(f)
# csv_writer.writerow(V)
# f.close()

V = np.array(V)
V = mat(V)  # 1×90的矩阵

result_numpy = np.matrix(result_numpy)
print(np.linalg.inv(result_numpy))
# 90×90的矩阵,result_numpy存储的是应变大小  V = result_numpy * a
# result_numpy.T * V = result_numpy.T * result_numpy * a
# (result_numpy.T * result_numpy).I * result_numpy.T * V = a


a = result_numpy.I * V.T  # result_numpy的逆和V的转置相乘。90×90 * 90×1 ==> 90×1
b = (result_numpy.T * result_numpy).I * result_numpy.T * V.T
print(a)
print(b)
