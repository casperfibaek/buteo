import numpy as np

def y_class(s):
    if s == "fid":
        return 0
    if s == "area" or s == "b_area":
        return 1
    if s == "volume" or s == "b_volume":
        return 2
    if s == "ppl" or s == "ppl_ha":
        return 3

def sar_class(s):
    if s == "asc" or s == "ascending":
        return 0
    if s == "desc" or s == "descending":
        return 1
    if s == "joined" or s == "max":
        return 2

def count_freq(arr):
    yy = np.bincount(arr)
    ii = np.nonzero(yy)[0]
    return np.vstack((ii, yy[ii])).T

# https://stackoverflow.com/a/44233061/8564588
def minority_class_mask(arr, minority):
    return np.hstack([
        np.random.choice(np.where(arr == l)[0], minority, replace=False) for l in np.unique(arr)
    ])

def get_shape(numpy_arr):
    return (numpy_arr.shape[1], numpy_arr.shape[2], numpy_arr.shape[3])

def add_rotations(X):
    return np.concatenate([
        np.rot90(X, k=1, axes=(1, 2)),
        np.rot90(X, k=2, axes=(1, 2)),
        np.rot90(X, k=3, axes=(1, 2)),
        X,
    ])

def add_noise(X, amount=0.01):
    return X * np.random.normal(1, amount, X.shape)

def scale_to_01(arr):
      return (arr - arr.min()) / (arr.max() - arr.min())

def decay(epoch):
  if epoch < 3:
    return 1e-3
  elif epoch >= 3 and epoch < 7:
    return 1e-4
  else:
    return 1e-5