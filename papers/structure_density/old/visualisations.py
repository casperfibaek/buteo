from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras import Sequential, Model
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, BatchNormalization, Concatenate, Input
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.models import load_model

import os
import ml_utils
import numpy as np

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
folder = "C:\\Users\\caspe\\Desktop\\Paper_2_StruturalDensity\\analysis\\"
size = 160
target = "area"

# ***********************************************************************
#                   LOADING DATA
# ***********************************************************************

blue = 0
green = 1
red = 2
nir = 0

# Load and scale RGB channels
X_rgb = np.load(folder + f"{str(int(size))}_rgb.npy").astype('float32')
X_rgb[:, :, :, blue] = ml_utils.scale_to_01(np.clip(X_rgb[:, :, :, blue], 0, 4000))
X_rgb[:, :, :, green] = ml_utils.scale_to_01(np.clip(X_rgb[:, :, :, green], 0, 5000))
X_rgb[:, :, :, red] = ml_utils.scale_to_01(np.clip(X_rgb[:, :, :, red], 0, 6000))

# Load and scale NIR channel (Add additional axis to match RGB)
X_nir = np.load(folder + f"{str(int(size))}_nir.npy").astype('float32')
X_nir = X_nir[:, :, :, np.newaxis]
X_nir[:, :, :, nir] = ml_utils.scale_to_01(np.clip(X_nir[:, :, :, nir], 0, 11000))

# Merge RGB and NIR
X = np.concatenate([X_rgb, X_nir], axis=3)

# Load Backscatter (asc + desc), clip the largest outliers (> 99%)
bs = np.load(folder + f"{str(int(size))}_bs.npy")[:, :, :, [ml_utils.sar_class("asc"), ml_utils.sar_class("desc")]]
bs = ml_utils.scale_to_01(np.clip(bs, 0, np.quantile(bs, 0.99)))
bs = np.concatenate([
    bs.mean(axis=(1,2)),
    bs.std(axis=(1,2)),
    bs.min(axis=(1,2)),
    bs.max(axis=(1,2)),
    np.median(bs, axis=(1,2)),
], axis=1)

# Load coherence
coh = np.load(folder + f"{str(int(size))}_coh.npy")[:, :, :, [ml_utils.sar_class("asc"), ml_utils.sar_class("desc")]]
coh = np.concatenate([
    coh.mean(axis=(1,2)),
    coh.std(axis=(1,2)),
    coh.min(axis=(1,2)),
    coh.max(axis=(1,2)),
    np.median(coh, axis=(1,2)),
], axis=1)

sar = np.concatenate([bs, coh], axis=1)

y = np.load(folder + f"{str(int(size))}_y.npy")[:, ml_utils.y_class(target)]
y = (size * size) * y # Small house (100m2 * 4m avg. height)

if target == "area":
    res_mult = 140
    labels = [*range(0, 5740, res_mult)]
else:
    res_mult = 700
    labels = [*range(0, 28700, res_mult)]

X_rgb = None
X_nir = None
bs = None
coh = None

# ***********************************************************************
#                   Visualisation DATA
# ***********************************************************************
import matplotlib.pyplot as plt

best_model = load_model(f"./models/best_model_area.hdf5")

truth = y.astype("float32")
predicted = best_model.predict([X, sar]).squeeze().astype("float32")

truth_labels = np.digitize(y, labels, right=True)
predicted_labels = np.digitize(predicted, labels, right=True)
labels_unique = np.unique(truth_labels)

residuals = ((truth - predicted) / res_mult).astype('float32')
median_absolute = np.median(np.abs(residuals))

fig1, ax = plt.subplots()
ax.set_title('Boxplot area')

per_class = []
for cl in labels_unique:
    per_class.append(residuals[truth_labels == cl])

ax.boxplot(per_class, showfliers=False)
ax.violinplot(per_class, showextrema=False, showmedians=True)

plt.show()
import pdb; pdb.set_trace()

