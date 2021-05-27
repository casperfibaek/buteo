import numpy as np
from utils import (
    preprocess_optical,
    preprocess_sar,
    random_scale_noise,
)
from glob import glob

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/machine_learning_data/"

# regions = [
#     1081,
#     1082,
#     1083,
#     1084,
#     1085,
# ]

leave_out = 461

rgbn_dir = []
swir_dir = []
sar_dir = []
area_dir = []
volume_dir = []
people_dir = []

rgbn_files = glob(folder + "*_RGBN.npy")
for rgbn_path in rgbn_files:
    if f"0{leave_out}" in rgbn_path:
        continue
    rgbn_dir.append(preprocess_optical(random_scale_noise(np.load(rgbn_path))))

rgbn_dir = np.concatenate(rgbn_dir)
np.save(folder + "000_RGBN.npy", rgbn_dir)
rgbn_dir = None

swir_files = glob(folder + "*_SWIR.npy")
for swir_path in swir_files:
    if f"0{leave_out}" in swir_path:
        continue
    swir_dir.append(preprocess_optical(random_scale_noise(np.load(swir_path))))

swir_dir = np.concatenate(swir_dir)
np.save(folder + "000_SWIR.npy", swir_dir)
swir_dir = None

sar_files = glob(folder + "*_SAR.npy")
for sar_path in sar_files:
    if f"0{leave_out}" in sar_path:
        continue
    preprocessed = preprocess_sar(random_scale_noise(np.load(sar_path)))
    sar_dir.append(preprocessed)


sar_dir = np.concatenate(sar_dir)
np.save(folder + "000_SAR.npy", sar_dir)
sar_dir = None

area_files = glob(folder + "*_LABEL_AREA.npy")
for area_path in area_files:
    if f"0{leave_out}" in area_path:
        continue
    area_dir.append(np.load(area_path))

area_dir = np.concatenate(area_dir)
np.save(folder + "000_LABEL_AREA.npy", area_dir)
area_dir = None

volume_files = glob(folder + "*_LABEL_VOLUME.npy")
for volume_path in volume_files:
    if f"0{leave_out}" in volume_path:
        continue
    volume_dir.append(np.load(volume_path))

volume_dir = np.concatenate(volume_dir)
np.save(folder + "000_LABEL_VOLUME.npy", volume_dir)
volume_dir = None

people_files = glob(folder + "*_LABEL_PEOPLE.npy")
for people_path in people_files:
    if f"0{leave_out}" in people_path:
        continue
    people_dir.append(np.load(people_path))

people_dir = np.concatenate(people_dir)
np.save(folder + "000_LABEL_PEOPLE.npy", people_dir)
people_dir = None

swir = np.load(folder + "000_SWIR.npy")

sample_arr = [True, False]
shuffle_mask = np.random.permutation(swir.shape[0])
flip = np.random.choice(sample_arr, size=swir.shape[0])
norm = ~flip
swir = None

for name in [
    "000_SWIR.npy",
    "000_RGBN.npy",
    "000_SAR.npy",
    "000_LABEL_AREA.npy",
    "000_LABEL_VOLUME.npy",
    "000_LABEL_PEOPLE.npy",
]:
    arr = np.load(folder + name)
    new_name = "001_" + name.split("_")[1]
    if name == "000_LABEL_AREA.npy":
        new_name = "001_LABEL_AREA.npy"
    elif name == "000_LABEL_VOLUME.npy":
        new_name = "001_LABEL_VOLUME.npy"
    elif name == "000_LABEL_PEOPLE.npy":
        new_name = "001_LABEL_PEOPLE.npy"

    np.save(
        folder + new_name,
        np.concatenate(
            [
                np.rot90(arr[flip], k=2, axes=(1, 2)),
                arr[norm],
            ]
        )[shuffle_mask],
    )
