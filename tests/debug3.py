""" Get simple EO data and labels for testing model architectures. """

import os
from glob import glob
import sys; sys.path.append("../")
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import buteo as beo


FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/projects/model_zoo/data/patches/"

x_train = beo.MultiArray([np.load(f, mmap_mode="r") for f in glob(os.path.join(FOLDER, "*train_s2.npy"))])
y_train = beo.MultiArray([np.load(f, mmap_mode="r") for f in glob(os.path.join(FOLDER, "*train_label_area.npy"))])

x_test = beo.MultiArray([np.load(f, mmap_mode="r") for f in glob(os.path.join(FOLDER, "*test_s2.npy"))])
y_test = beo.MultiArray([np.load(f, mmap_mode="r") for f in glob(os.path.join(FOLDER, "*test_label_area.npy"))])

assert len(x_train) == len(y_train)
assert len(x_test) == len(y_test)

def callback_normalise(x, y):
    x_norm = np.empty_like(x, dtype=np.float32)
    np.divide(x, 10000.0, out=x_norm)

    return x_norm, y

def callback_convert(x, y):
    return torch.from_numpy(x), torch.from_numpy(y)

def callback_norm_convert(x, y):
    x, y = callback_normalise(x, y)
    return callback_convert(x, y)

ds_train = beo.AugmentationDataset(
    x_train,
    y_train,
    input_is_channel_last=True,
    output_is_channel_last=False,
    callback_pre=callback_normalise,
    callback=callback_convert,
    augmentations=[
        { "name": "noise_uniform", "p": 0.2 },
        { "name": "rotation_xy", "p": 0.2 },
        { "name": "mirror_xy", "p": 0.2 },
        { "name": "cutmix", "p": 0.2 },
    ],
)
ds_test = beo.Dataset(x_test, y_test, input_is_channel_last=True, output_is_channel_last=False, callback=callback_norm_convert)

dl_train = DataLoader(ds_train, batch_size=32, shuffle=True, pin_memory=True, num_workers=0, drop_last=True)
dl_test = DataLoader(ds_test, batch_size=32, shuffle=False, pin_memory=True, num_workers=0)

for i, (inputs, targets) in enumerate(tqdm(dl_train, total=len(dl_train))):
    pass

import pdb; pdb.set_trace()
