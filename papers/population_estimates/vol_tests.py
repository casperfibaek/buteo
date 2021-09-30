import numpy as np
from numpy import genfromtxt

np.set_printoptions(suppress=True)

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/tmp/"
csv_file = folder + "buildings_with_volume.csv"
npy_file = folder + "buildings_with_volume.npy"

# vol_sum, hot_mean, area, perimeter, ipq, height, volume, hull_ratio
# 0,       1,        2,    3,         4,   5,      6,      7,

data = np.load(npy_file)
# data = genfromtxt(csv_file, delimiter=",", dtype="float32")

# # remove header
# data = data[1:]

# # remove fid
# data = data[:, 1:]

# remove buildings with an average height of less than 1.2m
vol_rat = data[:, 6] / data[:, 2]
height_mask = np.logical_or(data[:, 1] > 1.2, (data[:, 6] / data[:, 2]) < 1.2)
data = data[height_mask]

vol_rat = data[:, 6] / data[:, 2]
low = np.quantile(vol_rat, 0.05)
high = np.quantile(vol_rat, 0.95)
height_mask = np.logical_and(vol_rat > low, vol_rat < high)
data = data[height_mask]


vol_sum = data[:, 0]
hot_mean = data[:, 1]
area = data[:, 2]
perimeter = data[:, 3]
ipq = data[:, 4]
height = data[:, 5]
volume = data[:, 6]
hull_ratio = data[:, 7]

np.save(folder + "x_train.npy", data[:, [2, 3, 4, 7]])
np.save(folder + "y_train.npy", volume)

import pdb

pdb.set_trace()
