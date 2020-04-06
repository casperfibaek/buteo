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

# SCL
##  0: SC_NODATA
##  1: SC_SATURATED_DEFECTIVE
##  2: SC_DARK_FEATURE_SHADOW
##  3: SC_CLOUD_SHADOW
##  4: SC_VEGETATION
##  5: SC_NOT_VEGETATED
##  6: SC_WATER
##  7: SC_UNCLASSIFIED
##  8: SC_CLOUD_MEDIUM_PROBA
##  9: SC_CLOUD_HIGH_PROBA
## 10: SC_THIN_CIRRUS
## 11: SC_SNOW_ICE


cdef int assess_quality_func(
    int scl,
    int b1,
    int b2,
    int nodata,
) nogil:
    cdef int quality = 0

    if nodata == 1: # SC_NODATA
        quality = 0 
    elif scl == 1:  # SC_SATURATED_DEFECTIVE
        quality = 0 
    elif scl == 2:  # SC_DARK_FEATURE_SHADOW
        quality = 1
    elif scl == 3:  # SC_CLOUD_SHADOW
        quality = 1
    elif scl == 4:  # SC_VEGETATION
        quality = 10
    elif scl == 5:  # SC_NOT_VEGETATED
        quality = 10
    elif scl == 6:  # SC_WATER
        quality = 9
    elif scl == 7:  # SC_UNCLASSIFIED
        quality = 9
    elif scl == 8:  # SC_CLOUD_MEDIUM_PROBA
        quality = 3
    elif scl == 9:  # SC_CLOUD_HIGH_PROBA
        quality = 1
    elif scl == 10: # SC_THIN_CIRRUS
        quality = 8
    elif scl == 11: # SC_SNOW_ICE
        quality = 6
    
    # land pixels
    if scl == 4 or scl == 7:
        if b1 <= 1 and b2 <= 1:
            quality -= 4
        elif b1 <= 10 and b2 <= 30:
            quality -= 3
        elif b1 <= 40 and b2 <= 70:
            quality -= 2
        elif b1 <= 80 and b2 <= 125:
            quality -= 1

        elif b1 > 1400:
            quality -= 3
        elif b1 > 1100:
            quality -= 2
        elif b1 > 800:
            quality -= 1
    
    # non vegetation (often bare soil or urban)
    if scl == 5:
        if b1 <= 1 and b2 <= 1:
            quality -= 4
        elif b1 <= 10 and b2 <= 30:
            quality -= 3
        elif b1 <= 40 and b2 <= 70:
            quality -= 2
        elif b1 <= 80 and b2 <= 125:
            quality -= 1

        elif b1 > 1500:
            quality -= 3
        elif b1 > 1200:
            quality -= 2
        elif b1 > 900:
            quality -= 1    
    
    # water pixels
    elif scl == 6:
        if b1 <= 50 or b2 <= 100:
            quality -= 1
        elif b1 > 900:
            quality -= 3
        elif b1 > 700:
            quality -= 2
        elif b1 > 500:
            quality -= 1
    
    # cirrus
    elif scl == 10:
        if b1 <= 1 and b2 <= 1:
            quality -= 4
        elif b1 <= 10 and b2 <= 50:
            quality -= 3
        elif b1 <= 50 and b2 <= 100:
            quality -= 2
        elif b1 <= 85 and b2 <= 125:
            quality -= 1

        elif b1 > 1350:
            quality -= 3
        elif b1 > 1000:
            quality -= 2
        elif b1 > 700:
            quality -= 1   
    
    # other
    else:
        if b1 > 1000:
            quality -= 1

    if quality <= 0:
        if nodata == 1 or scl == 1:
            quality = 0
        else:
            quality = 1

    return quality


cdef void assess_quality(
    int [:, ::1] scl,
    int [:, ::1] b1,
    int [:, ::1] b2,
    int [:, ::1] nodata,
    int [:, ::1] quality,
    int x_max,
    int y_max,
) nogil:
    cdef int x, y

    for x in prange(x_max, nogil=True):
        for y in prange(y_max):
            quality[x][y] = assess_quality_func(scl[x][y], b1[x][y], b2[x][y], nodata[x][y])


cpdef void radiometric_quality(scl, b1, b2, nodata, quality):
    cdef int [:, ::1] scl_view = scl
    cdef int [:, ::1] b1_view = b1
    cdef int [:, ::1] b2_view = b2
    cdef int [:, ::1] nodata_view = nodata
    cdef int [:, ::1] quality_view = quality

    cdef int x_max = scl.shape[0]
    cdef int y_max = scl.shape[1]
    
    assess_quality(scl_view, b1_view, b2_view, nodata_view, quality_view, x_max, y_max)


cdef int assess_quality_spatial_func(
    int scl,
    int quality,
    double distance,
    int quality_eroded,
) nogil:
    cdef int negative = 0
    # cdef double float_negative = (-0.02 * distance) + 3

    if distance < 25 == 1:
        negative += 3
    elif distance < 50:
        negative += 2
    elif distance < 100:
        negative += 1
    
    if (quality - negative) < quality_eroded:
        return quality_eroded
    else:
        return quality - negative

cdef int assess_quality_spatial(
    int [:, ::1] scl,
    int [:, ::1] quality,
    double [:, ::1] distance,
    int [:, ::1] quality_eroded,
    bint score,
    int x_max,
    int y_max,
) nogil:
    cdef int x, y, q
    cdef int score_sum = 0
    for x in prange(x_max, nogil=True):
        for y in prange(y_max):
            q = assess_quality_spatial_func(scl[x][y], quality[x][y], distance[x][y], quality_eroded[x][y])
            quality[x][y] = q
            score_sum += q
    
    return score_sum


cpdef int radiometric_quality_spatial(scl, quality, distance, quality_eroded, score):
    cdef int [:, ::1] scl_view = scl
    cdef int [:, ::1] quality_view = quality
    cdef double [:, ::1] distance_view = distance
    cdef int [:, ::1] quality_eroded_view = quality_eroded
    cdef bint bint_score = score

    cdef int x_max = scl.shape[0]
    cdef int y_max = scl.shape[1]

    return assess_quality_spatial(scl_view, quality_view, distance_view, quality_eroded_view, bint_score, x_max, y_max)


cdef void assess_quality_haze(
    int [:, ::1] previous_quality,
    int [:, ::1] previous_b1,
    int [:, ::1] quality,
    int [:, ::1] b1,
    int [:, ::1] haze_change,
    int x_max,
    int y_max,
) nogil:
    cdef int x, y
    cdef double ratio
    for x in prange(x_max, nogil=True):
        for y in prange(y_max):
            if previous_quality[x][y] <= 0:
                if quality[x][y] > 0:
                    haze_change[x][y] = 1
            elif quality[x][y] >= previous_quality[x][y]:
                if previous_b1[x][y] == 0:
                    haze_change[x][y] = 1
                else:
                    ratio = b1[x][y] / previous_b1[x][y]
                    if ratio >= 2.0:
                        if quality[x][y] >= 3:
                            quality[x][y] = quality[x][y] - 2
                        elif quality[x][y] >= 2:
                            quality[x][y] = quality[x][y] - 1
                    elif ratio >= 1.1:
                        if quality[x][y] >= 2:
                            quality[x][y] = quality[x][y] - 1
                    elif ratio <= 0.9:
                        haze_change[x][y] = 1



cpdef void radiometric_quality_haze(previous_quality, previous_b1, quality, b1, haze_change):
    cdef int [:, ::1] previous_quality_view = previous_quality
    cdef int [:, ::1] previous_b1_view = previous_b1
    cdef int [:, ::1] quality_view = quality
    cdef int [:, ::1] b1_view = b1
    
    cdef int [:, ::1] haze_change_view = haze_change

    cdef int x_max = quality.shape[0]
    cdef int y_max = quality.shape[1]
    
    assess_quality_haze(previous_quality_view, previous_b1_view, quality_view, b1_view, haze_change_view, x_max, y_max)


cdef void assess_radiometric_change_mask(
    int [:, ::1] previous_quality,
    int [:, ::1] quality,
    int [:, ::1] previous_scl,
    int [:, ::1] scl,
    int [:, ::1] haze,
    int [:, ::1] result,
    int x_max,
    int y_max,
) nogil:
    cdef int x, y
    for x in prange(x_max, nogil=True):
        for y in prange(y_max):
            if quality[x][y] > previous_quality[x][y]:
                result[x][y] = 1
            elif quality[x][y] == previous_quality[x][y] and (scl[x][y] == 4 or scl[x][y] == 5 or scl[x][y] == 6 or scl[x][y] == 7):
                if haze[x][y] == 1:
                    result[x][y] = 1


cpdef int [:, ::1] radiometric_change_mask(previous_quality, quality, previous_scl, scl, haze, result):
    cdef int [:, ::1] previous_quality_view = previous_quality
    cdef int [:, ::1] quality_view = quality
    cdef int [:, ::1] previous_scl_view = previous_scl
    cdef int [:, ::1] scl_view = scl
    cdef int [:, ::1] haze_view = haze
    cdef int [:, ::1] result_view = result
    cdef int x_max = quality.shape[0]
    cdef int y_max = quality.shape[1]
    
    assess_radiometric_change_mask(
        previous_quality_view,
        quality_view,
        previous_scl,
        scl_view,
        haze_view,
        result_view,
        x_max,
        y_max,
    )

    return result
