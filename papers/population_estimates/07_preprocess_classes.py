import numpy as np
from scipy import stats

folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/ghana/vector/classes/patches/merged/"

label = np.load(folder + "class_label_area.npy")

shaped = label.reshape(
    label.shape[0], (label.shape[1] * label.shape[2] * label.shape[3])
)

mode = stats.mode(shaped, axis=1)
mode = mode.mode[:, 0]

merge = np.empty((mode.shape[0], 2), dtype=int)
merge[:, 0] = mode
merge[:, 1] = np.arange(mode.shape[0])

class_0 = merge[merge[:, 0] == 0][:5000]
class_1 = merge[merge[:, 0] == 1][:5000]
class_2 = merge[merge[:, 0] == 2]
class_2 = np.concatenate([class_2, class_2, class_2, class_2, class_2, class_2])[:5000]
class_3 = merge[merge[:, 0] == 3]
class_3 = np.concatenate([class_3, class_3, class_3, class_3, class_3, class_3])[:5000]

merged = np.concatenate(
    [
        class_0,
        class_1,
        class_2,
        class_3,
    ]
)

mask = merged[:, 1]
shuffle_mask = np.random.permutation(mask.shape[0])
mask = mask[shuffle_mask]

np.save(
    folder + "class_balanced_label_area.npy",
    np.load(folder + "class_label_area.npy")[mask],
)
np.save(folder + "class_balanced_RGBN.npy", np.load(folder + "class_RGBN.npy")[mask])
np.save(folder + "class_balanced_SAR.npy", np.load(folder + "class_SAR.npy")[mask])
np.save(
    folder + "class_balanced_RESWIR.npy", np.load(folder + "class_RESWIR.npy")[mask]
)
