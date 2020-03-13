# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, profile = False
cimport cython
from cython.parallel cimport prange
import numpy as np


cdef void truncate_2d(
    double [:, ::1] arr,
    double [:, ::1] result,
    double min_value,
    double max_value,
    int x_max,
    int y_max,
    bint has_nodata,
    double fill_value,
) nogil:
    cdef int x, y
    for x in prange(x_max):
        for y in range(y_max):
            if has_nodata is True and arr[x][y] == fill_value:
                result[x][y] = max_value
            elif arr[x][y] > max_value:
                result[x][y] = max_value
            elif arr[x][y] < min_value:
                result[x][y] = min_value
            else:
                result[x][y] = arr[x][y]


cdef void truncate_3d(
    double [:, :, ::1] arr,
    double [:, :, ::1] result,
    double min_value,
    double max_value,
    int x_max,
    int y_max,
    int z_max,
    bint has_nodata,
    double fill_value,
) nogil:
    cdef int x, y, z
    for x in prange(x_max):
        for y in range(y_max):
            for z in range(z_max):
                if has_nodata is True and arr[x][y][z] == fill_value:
                    result[x][y][z] = fill_value
                elif arr[x][y][z] > max_value:
                    result[x][y][z] = max_value
                elif arr[x][y][z] < min_value:
                    result[x][y][z] = min_value
                else:
                    result[x][y][z] = arr[x][y][z]


cdef void threshold_2d(
  double [:, ::1] arr,
  int [:, ::1] result,
  double min_value,
  double max_value,
  int x_max,
  int y_max,
  bint has_nodata,
  double fill_value,
  bint invert,
) nogil:
    cdef int x, y, p, f
    p = 1
    f = 0
    if invert is True:
        p = 0
        f = 1
        
    for x in prange(x_max):
        for y in range(y_max):
            if has_nodata is True and arr[x][y] == fill_value:
                result[x][y] = fill_value
            elif arr[x][y] <= max_value and arr[x][y] >= min_value:
                result[x][y] = p
            else:
                result[x][y] = f


cdef void threshold_3d(
  double [:, :, ::1] arr,
  int [:, :, ::1] result,
  double min_value,
  double max_value,
  int x_max,
  int y_max,
  int z_max,
  bint has_nodata,
  double fill_value,
  bint invert,
) nogil:
    cdef int x, y, z, p, f
    p = 1
    f = 0
    if invert is True:
        p = 0
        f = 1
    for x in prange(x_max):
        for y in range(y_max):
            for z in range(z_max):
                if has_nodata is True and arr[x][y][z] == fill_value:
                    result[x][y][z] = fill_value
                if arr[x][y][z] <= max_value and arr[x][y][z] >= min_value:
                    result[x][y][z] = p
                else:
                    result[x][y][z] = f


def threshold_array(arr, min_value=False, max_value=False, invert=False):
    cdef bint has_nodata = False
    cdef double fill_value
    cdef double min_v = min_value if min_value is not False else DBL_MIN
    cdef double max_v = max_value if max_value is not False else DBL_MAX
    cdef bint inv = invert
    cdef double [:, ::1] arr_view_2d
    cdef double [:, :, ::1] arr_view_3d
    cdef int [:, ::1] result_view_2d
    cdef int [:, :, ::1] result_view_3d
    cdef int dims = len(arr.shape)

    assert(dims == 2 or dims == 3)

    arr = arr.astype(np.double) if arr.dtype != np.double else arr
    result = np.zeros(arr.shape, dtype=np.intc)

    if isinstance(arr, np.ma.MaskedArray):
        result = np.ma.array(result, fill_value=arr.fill_value)
        fill_value = arr.fill_value
        has_nodata = 1
        arr = arr.filled()
    else:
        fill_value = 0.0

    if dims == 3:
        result_view_3d = result
        arr_view_3d = arr
        threshold_3d(
            arr_view_3d,
            result_view_3d,
            min_v,
            max_v,
            arr.shape[0],
            arr.shape[1],
            arr.shape[2],
            has_nodata,
            fill_value,
            invert,
        )
    else:
        arr_view_2d = arr
        result_view_2d = result
        threshold_2d(
            arr_view_2d,
            result_view_2d,
            min_v,
            max_v,
            arr.shape[0],
            arr.shape[1],
            has_nodata,
            fill_value,
            invert,
        )
    
    if isinstance(arr, np.ma.MaskedArray):
        return np.ma.masked_where(result == fill_value, result).astype('uint8')

    return result.astype('uint8')


def truncate_array(arr, min_value=False, max_value=False):
    cdef bint has_nodata = False
    cdef double fill_value
    cdef double min_v = min_value if min_value is not False else DBL_MIN
    cdef double max_v = max_value if max_value is not False else DBL_MAX
    cdef double [:, ::1] arr_view_2d
    cdef double [:, :, ::1] arr_view_3d
    cdef double [:, ::1] result_view_2d
    cdef double [:, :, ::1] result_view_3d
    cdef int dims = len(arr.shape)

    assert(dims == 2 or dims == 3)

    result = np.empty(arr.shape, dtype=np.double)
    arr = arr.astype(np.double) if arr.dtype != np.double else arr

    if isinstance(arr, np.ma.MaskedArray):
        result = np.ma.array(result, fill_value=arr.fill_value)
        fill_value = arr.fill_value
        has_nodata = 1
        arr = arr.filled()
    else:
        fill_value = 0.0

    if dims == 3:
        arr_view_3d = arr
        result_view_3d = result
        truncate_3d(
            arr_view_3d,
            result_view_3d,
            min_v,
            max_v,
            arr.shape[0],
            arr.shape[1],
            arr.shape[2],
            has_nodata,
            fill_value,
        )
    else:
        arr_view_2d = arr
        result_view_2d = result
        truncate_2d(
            arr_view_2d,
            result_view_2d,
            min_v,
            max_v,
            arr.shape[0],
            arr.shape[1],
            has_nodata,
            fill_value,
        )
    
    if isinstance(arr, np.ma.MaskedArray):
        return np.ma.masked_where(result == fill_value, result)

    return result

