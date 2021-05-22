import matplotlib.pyplot as plt
import numpy
import csv
import numpy as np
import json

# def manage_information(all_information,location):
#     num=0
#     for item in all_information:
#         for loc in location:
#             if float(item[0]) in loc and round(float(item[2]),2) in loc:
#

with open('loc_num.json', 'r') as f:
    location = json.load(f)
# print(location)
f.close()
with open('output_data/all_information1.csv') as f:
    forcedata = f.readlines()
print(len(forcedata))
f.close()
all_information=[]
for num, item in enumerate(forcedata):
    if num == 0: continue
    data = item.split(',')
    all_information.append(data)
    numpy1=np.zeros((3,15))
    numpy2=np.zeros_like(numpy1)
    if len(all_information) == 90:
        manage_information(all_information,location)
        break