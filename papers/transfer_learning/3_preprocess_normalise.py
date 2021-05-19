import numpy as np
from utils import (
    preprocess_optical,
    preprocess_sar,
    random_scale_noise,
    rotate_shuffle,
)

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/machine_learning_data/"

municipalities = np.load(folder + "municipalities.npy")


test_munis = [
    101,  # copenhagen
    147,  # frederiksberg
    185,  # taarnby
    155,  # dragoer
    461,  # odense
    390,  # vordingborg
    657,  # herning
    707,  # norddjurs
    630,  # vejle
    411,  # christiansoe
    250,  # frederiksund
    550,  # toender
]

rgbn_dir = []
swir_dir = []
sar_dir = []
labels_dir = []


for muni in municipalities:
    if muni not in test_munis:
        continue

    rgbn = preprocess_optical(random_scale_noise(np.load(folder + f"{muni}_RGBN.npy")))
    swir = preprocess_optical(random_scale_noise(np.load(folder + f"{muni}_SWIR.npy")))
    sar = preprocess_sar(random_scale_noise(np.load(folder + f"{muni}_SAR.npy")))
    labels = np.load(folder + f"{muni}_LABELS.npy")

    rgbn, swir, sar, labels = rotate_shuffle([rgbn, swir, sar, labels])

    rgbn_dir.append(rgbn)
    swir_dir.append(swir)
    sar_dir.append(sar)
    labels_dir.append(labels)

np.save(folder + "000_RGBN", np.concatenate(rgbn_dir))
np.save(folder + "000_SWIR", np.concatenate(swir_dir))
np.save(folder + "000_SAR", np.concatenate(sar_dir))
np.save(folder + "000_LABELS", np.concatenate(labels_dir))
