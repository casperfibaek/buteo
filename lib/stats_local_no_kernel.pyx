# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, profile = False
cimport cython
from cython.parallel cimport prange
from libc.math cimport fabs, exp
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
    for z in range(z_max):
        for x in prange(x_max):
            for y in range(y_max):
                if has_nodata is True and arr[z][x][y] == fill_value:
                    result[z][x][y] = fill_value
                elif arr[z][x][y] > max_value:
                    result[z][x][y] = max_value
                elif arr[z][x][y] < min_value:
                    result[z][x][y] = min_value
                else:
                    result[z][x][y] = arr[z][x][y]


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
                result[x][y] = <int>fill_value
            elif arr[x][y] <= max_value and arr[x][y] >= min_value:
                result[x][y] = p
            else:
                result[x][y] = f


cdef void threshold_3d(
  double [:, :, ::1] arr,
  int [:, :, ::1] result,
  double min_value,
  double max_value,
  int z_max,
  int x_max,
  int y_max,
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
    for z in range(z_max):
        for x in prange(x_max):
            for y in range(y_max):
                if has_nodata is True and arr[z][x][y] == fill_value:
                    result[z][x][y] = <int>fill_value
                if arr[x][y][z] <= max_value and arr[z][x][y] >= min_value:
                    result[z][x][y] = p
                else:
                    result[z][x][y] = f



cdef double cdf(double z) nogil:
    cdef double t = 1 / (1 + .2315419 * fabs(z))
    cdef double d =.3989423 * exp( -z * z / 2)
    cdef double prob = d * t * (.3193815 + t * ( -.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    
    if( z > 0 ) :
        prob = 1 - prob
        
    return 1.0 - ((prob - 0.5) / 0.5)


cdef void cdf_2d(
  double [:, ::1] arr,
  double [:, ::1] result,
  int x_max,
  int y_max,
  bint has_nodata,
  double fill_value,
) nogil:
    cdef int x, y
        
    for x in prange(x_max):
        for y in range(y_max):
            if has_nodata is True and arr[x][y] == fill_value:
                result[x][y] = fill_value
            else:
                result[x][y] = cdf(arr[x][y])


cdef void cdf_3d(
  double [:, :, ::1] arr,
  double [:, :, ::1] result,
  int z_max,
  int x_max,
  int y_max,
  bint has_nodata,
  double fill_value,
) nogil:
    cdef int x, y, z
    for x in prange(x_max):
        for y in range(y_max):
            for z in range(z_max):
                if has_nodata is True and arr[z][x][y] == fill_value:
                    result[z][x][y] = fill_value
                else:
                    result[z][x][y] = cdf(arr[z][x][y])


def cdef_from_z(arr):
    cdef bint has_nodata = False
    cdef double fill_value
    cdef double [:, ::1] arr_view_2d
    cdef double [:, :, ::1] arr_view_3d
    cdef double [:, ::1] result_view_2d
    cdef double [:, :, ::1] result_view_3d
    cdef int dims = len(arr.shape)

    assert(dims == 2 or dims == 3)

    result = np.empty(arr.shape, dtype=np.double)
    arr = arr.astype(np.double) if arr.dtype != np.double else arr
    arr_mask = False

    if isinstance(arr, np.ma.MaskedArray):
        result = np.ma.array(result, fill_value=arr.fill_value)
        fill_value = arr.fill_value
        has_nodata = 1
        if arr.mask is not False:
            arr_mask = np.ma.getmask(arr)
        arr = arr.filled()
    else:
        fill_value = 0.0

    if dims == 3:
        arr_view_3d = arr
        result_view_3d = result
        cdf_3d(
            arr_view_3d,
            result_view_3d,
            arr.shape[0],
            arr.shape[1],
            arr.shape[2],
            has_nodata,
            fill_value,
        )
    else:
        arr_view_2d = arr
        result_view_2d = result
        cdf_2d(
            arr_view_2d,
            result_view_2d,
            arr.shape[0],
            arr.shape[1],
            has_nodata,
            fill_value,
        )
    
    if has_nodata is True:
        if arr_mask is not False:
            return np.ma.masked_where(arr_mask, result)
        else:
            return np.ma.masked_equal(result, fill_value)

    return result


def threshold_array(arr, min_value=False, max_value=False, invert=False):
    cdef bint has_nodata = False
    cdef double fill_value
    cdef double min_v = min_value if min_value is not False else -99999999999999999999.0
    cdef double max_v = max_value if max_value is not False else 99999999999999999999.0
    cdef bint inv = invert
    cdef double [:, ::1] arr_view_2d
    cdef double [:, :, ::1] arr_view_3d
    cdef int [:, ::1] result_view_2d
    cdef int [:, :, ::1] result_view_3d
    cdef int dims = len(arr.shape)

    assert(dims == 2 or dims == 3)

    arr = arr.astype(np.double) if arr.dtype != np.double else arr
    result = np.zeros(arr.shape, dtype=np.intc)
    arr_mask = False

    if isinstance(arr, np.ma.MaskedArray):
        result = np.ma.array(result, fill_value=arr.fill_value)
        fill_value = arr.fill_value
        has_nodata = 1
        if arr.mask is not False:
            arr_mask = np.ma.getmask(arr)
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
    
    if has_nodata is True:
        if arr_mask is not False:
            return_arr = np.ma.masked_where(arr_mask, result).set_fill_value(255)
            return return_arr.astype('uint8')
        else:
            return_arr = np.ma.masked_equal(result, fill_value).set_fill_value(255)
            return return_arr

    return result.astype('uint8')


def truncate_array(arr, min_value=False, max_value=False):
    cdef bint has_nodata = False
    cdef double fill_value
    cdef double min_v = min_value if min_value is not False else -99999999999999999999.0
    cdef double max_v = max_value if max_value is not False else 99999999999999999999.0
    cdef double [:, ::1] arr_view_2d
    cdef double [:, :, ::1] arr_view_3d
    cdef double [:, ::1] result_view_2d
    cdef double [:, :, ::1] result_view_3d
    cdef int dims = len(arr.shape)

    assert(dims == 2 or dims == 3)

    result = np.empty(arr.shape, dtype=np.double)
    arr = arr.astype(np.double) if arr.dtype != np.double else arr
    arr_mask = False

    if isinstance(arr, np.ma.MaskedArray):
        result = np.ma.array(result, fill_value=arr.fill_value)
        fill_value = arr.fill_value
        has_nodata = 1
        if arr.mask is not False:
            arr_mask = np.ma.getmask(arr)
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
    
    if has_nodata is True:
        if arr_mask is not False:
            return np.ma.masked_where(arr_mask, result)
        else:
            return np.ma.masked_equal(result, fill_value)

    return result


cdef void highest(
  double [:, :, ::1] arr,
  double [:, ::1] result,
  double [::1] weights,
  int z_max,
  int x_max,
  int y_max,
  int weight_len,
  bint has_nodata,
  double fill_value,
) nogil:
    cdef int x, y, z, i, nodata
    cdef double max_value
    cdef double value

    for x in prange(x_max):
        for y in range(y_max):
            nodata = 0
            i = 0
            max_value = -9999.9
            for z in range(z_max):
                if has_nodata is True and arr[z][x][y] == fill_value:
                    nodata = 1
                    break
                else:
                    value = arr[z][x][y] * weights[z]
                    if value > max_value:
                        i = (z + 1)
                        max_value = value
            if nodata == 1:
                result[x][y] = fill_value
            else:
                result[x][y] = i


def select_highest(arr, weights):
    cdef bint has_nodata = False
    cdef double fill_value
    cdef double [:, :, ::1] arr_view
    cdef double [:, ::1] result_view
    cdef double [::1] weights_view
    cdef int dims = len(arr.shape)

    assert(dims == 2 or dims == 3)

    result = np.empty((arr.shape[1], arr.shape[2]), dtype=np.double)
    arr = arr.astype(np.double) if arr.dtype != np.double else arr
    weights = weights.astype(np.double) if weights.dtype != np.double else weights

    arr_mask = False

    if isinstance(arr, np.ma.MaskedArray):
        result = np.ma.array(result, fill_value=arr.fill_value)
        fill_value = arr.fill_value
        has_nodata = 1
        if arr.mask is not False:
            arr_mask = np.ma.getmask(arr)
        arr = arr.filled()
    else:
        fill_value = 0.0

    arr_view = arr
    result_view = result
    weights_view = weights

    highest(
        arr_view,
        result_view,
        weights_view,
        arr.shape[0],
        arr.shape[1],
        arr.shape[2],
        int(len(weights)),
        has_nodata,
        fill_value,
    )
    
    if has_nodata is True:
        if arr_mask is not False:
            return np.ma.masked_where(arr_mask[0], result)
        else:
            return np.ma.masked_equal(result, fill_value)

    return result


