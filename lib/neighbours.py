import time
import numpy as np
from raster_to_array import raster_to_array
from array_to_raster import array_to_raster
import numba
from numba import jit

before = time.time()

b8 = 'D:\\pythonScripts\\yellow\\raster\\T32VNJ_20180727T104021_vis_pca_10m.tif'
b8_arr = raster_to_array(b8)


@numba.jit(nopython=True, parallel=True, fastmath=True)
def rolling_window(a, shape):
    s = (a.shape[0] - shape[0] + 1,) + (a.shape[1] - shape[1] + 1,) + shape
    strides = a.strides + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=s, strides=strides)


@numba.jit(nopython=True, parallel=True, fastmath=True)
def create_view(arr, shape=(3, 3)):
    r_extra = int(np.floor(shape[0] / 2))
    c_extra = int(np.floor(shape[1] / 2))
    out = np.empty((arr.shape[0] + 2 * r_extra, arr.shape[1] + 2 * c_extra))
    out[:] = np.nan
    out[r_extra:-r_extra, c_extra:-c_extra] = arr
    view = rolling_window(out, shape)
    return view


@numba.jit(nopython=True, parallel=True, fastmath=True)
def _local_statistics(raster_arr, radius=1):
    if radius == 1:
        view_shape = (3, 3)
    else:
        view_size = (3 * radius) - 1
        view_shape = (view_size, view_size)

    view = create_view(raster_arr, view_shape)

    new_array = np.empty(raster_arr.shape)

    for index, value in np.ndenumerate(raster_arr):
        new_array[index] = np.nanstd(view[index])

    return new_array


def local_statistics(raster, radius=1):
    raster_arr = raster_to_array(raster)
    return _local_statistics(raster_arr, radius)

rad3 = local_statistics(b8, radius=3)
rad2 = local_statistics(b8, radius=2)
rad1 = local_statistics(b8, radius=1)

avg_std = np.divide(np.add(np.add(rad3, rad2), rad1), 3)

array_to_raster(avg_std.astype('float32'), reference_raster=b8, out_raster='D:\\pythonScripts\\yellow\\raster\\T32VNJ_20180727T104021_vis_pca_10m_avg.tif')


print(f'execution took: {round(time.time() - before, 2)}s')
