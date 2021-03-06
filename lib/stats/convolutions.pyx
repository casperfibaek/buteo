# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, profile = False
cimport cython
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, pow, fabs

import numpy as np
from kernel_generator import generate_offsets

cdef extern from "<float.h>":
    const float FLT_MAX
    const float FLT_MIN


cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size, int(*compar)(const_void *, const_void *)) nogil


cdef struct Neighbourhood:
    float value
    float weight


cdef struct Offset:
  int x
  int y
  int z
  double weight


ctypedef float (*f_type) (Neighbourhood *, int) nogil


cdef int compare(const_void *a, const_void *b) nogil:
    cdef double v = (<Neighbourhood*> a).value - (<Neighbourhood*> b).value
    if v < 0: return -1
    if v > 0: return 1
    return 0


cdef double sum_of_weights(Neighbourhood * neighbourhood, int non_zero) nogil:
    cdef int x
    cdef double accum

    accum = 0
    for x in range(non_zero):
        accum += neighbourhood[x].weight

    return accum


cdef void normalise_neighbourhood(Neighbourhood * neighbourhood, int non_zero, float quant) nogil:
    cdef int x
    cdef double sum_of_weights = sum_of_weights(neighbourhood, non_zero)

    for x in range(non_zero):
        neighbourhood[x].weight = (neighbourhood[x].weight / sum_of_weights)


# TODO: Refactor mode for a neighbourhood approach.
cdef int n_mode(int * a, int n, float quant) nogil:
    cdef int i, j, count
    cdef int max_value = 0
    cdef int max_count = 0

    for i in range(n):
        count = 0

        for j in range(n):
            if a[j] == a[i]:
                count += 1

            if count > n // 2:
                max_count = count
                max_value = a[i]
                break
        
        if count > max_count:
            max_count = count
            max_value = a[i]
    
    return max_value


cdef double n_sum(Neighbourhood * neighbourhood, int non_zero, float quant) nogil:
    cdef int x
    cdef double accum

    accum = 0
    for x in range(non_zero):
        accum += neighbourhood[x].value * neighbourhood[x].weight

    return accum


cdef double n_deviations(Neighbourhood * neighbourhood, int non_zero, int power, float quant) nogil:
    cdef int x, y
    cdef double accum, weighted_average, deviations
    cdef double sum_of_weights = sum_of_weights(neighbourhood, non_zero)

    weighted_average = n_sum(neighbourhood, non_zero)

    deviations = 0
    for x in range(non_zero):
        deviations += neighbourhood[x].weight * (pow((neighbourhood[x].value - weighted_average), power))

    return deviations / sum_of_weights


cdef double n_quantile(Neighbourhood * neighbourhood, int non_zero, float quant) nogil:
    cdef double weighted_quantile, top, bot, tb
    cdef int i

    cdef double sum_of_weights = sum_of_weights(neighbourhood, non_zero)
    cdef double cumsum = 0
    cdef double* w_cum = <double*> malloc(sizeof(double) * non_zero)

    qsort(<void*> neighbourhood, non_zero, sizeof(Neighbourhood), compare)
  
    weighted_quantile = neighbourhood[non_zero - 1].value
    for i in range(non_zero):
        cumsum += neighbourhood[i].weight
        w_cum[i] = (cumsum - (quant * neighbourhood[i].weight)) / sum_of_weights

        if cumsum >= quant:
            if i == 0 or w_cum[i] == quant:
                weighted_quantile = neighbourhood[i].value
                break
                
            top = w_cum[i] - quant
            bot = quant - w_cum[i - 1]
            tb = top + bot

            top = 1 - (top / tb)
            bot = 1 - (bot / tb)

            weighted_quantile = (neighbourhood[i - 1].value * bot) + (neighbourhood[i].value * top)
            break

    free(w_cum)
    return weighted_quantile


cdef double n_max(Neighbourhood * neighbourhood, int non_zero, float quant) nogil:
    cdef int x, y, current_max_i
    cdef double current_max, val

    current_max = FLT_MIN

    if neighbourhood[0].weight != 0:
        current_max = neighbourhood[0].value / neighbourhood[0].weight

    current_max_i = 0
    for x in range(non_zero):
        val = neighbourhood[x].value * neighbourhood[x].weight
        if val > current_max:
            current_max = val
            current_max_i = x

    return neighbourhood[current_max_i].value * neighbourhood[current_max_i].weight


cdef double n_min(Neighbourhood * neighbourhood, int non_zero, float quant) nogil:
    cdef int x, y, current_min_i
    cdef double current_min

    current_min = FLT_MAX

    if neighbourhood[0].weight != 0:
        current_min = neighbourhood[0].value / neighbourhood[0].weight

    current_min_i = 0
    for x in range(non_zero):
        if current_min != 0:
            val = neighbourhood[x].value / neighbourhood[x].weight
            if val < current_min:
                current_min = val
                current_min_i = x

    return neighbourhood[current_min_i].value / neighbourhood[current_min_i].weight


cdef double n_variance(Neighbourhood * neighbourhood, int non_zero, float quant) nogil:
    cdef double variance = n_deviations(neighbourhood, non_zero, 2)
    return variance


cdef double n_mad(Neighbourhood * neighbourhood, int non_zero, float quant) nogil:
    cdef double weighted_median = n_quantile(neighbourhood, non_zero, float 0.5)
    cdef Neighbourhood * deviations = <Neighbourhood*> malloc(sizeof(Neighbourhood) * non_zero)

    for x in range(non_zero):
        deviations[x].value = fabs(neighbourhood[x].value - weighted_median)
        deviations[x].weight = neighbourhood[x].weight

    cdef double mad = neighbourhood_weighted_median(deviations, non_zero)

    free(deviations)

    return mad


cdef double n_skew(Neighbourhood * neighbourhood, int non_zero, float quant) nogil:
    cdef double standard_deviation = sqrt(n_variance(neighbourhood, non_zero))

    if standard_deviation == 0:
        return 0

    cdef double variance_3 = n_deviations(neighbourhood, non_zero, 3)

    return variance_3 / (pow(standard_deviation, 3))


cdef double n_kurtosis(Neighbourhood * neighbourhood, int non_zero, float quant) nogil:
    cdef double standard_deviation = sqrt(n_variance(neighbourhood, non_zero))

    if standard_deviation == 0:
        return 0

    cdef double variance_4 = n_deviations(neighbourhood, non_zero, 4)

    return (variance_4 / (pow(standard_deviation, 4))) - 3


cdef Offset * generate_offsets(double [:, :, ::1] kernel, int x_kernel_size, int y_kernel_size, int z_kernel_size, int non_zero) nogil:
    cdef int x, y, z
    
    cdef int radius_x = <int>(x_kernel_size / 2)
    cdef int radius_y = <int>(y_kernel_size / 2)
    cdef int radius_z = <int>(z_kernel_size / 2)

    cdef int step = 0

    cdef Offset *offsets = <Offset *> malloc(non_zero * sizeof(Offset))

    for x in range(x_kernel_size):
        for y in range(y_kernel_size):
            for z in range(z_kernel_size):
                if kernel[x, y, z] != 0.0:
                    offsets[step].x = x - radius_x
                    offsets[step].y = y - radius_y
                    offsets[step].z = z - radius_z
                    offsets[step].weight = kernel[x, y, z]
                    step += 1

    return offsets


cdef f_type func_selector(str func_type):
    if func_type is 'mean': return n_sum
    elif func_type is 'dilate': return n_max
    elif func_type is 'erode': return n_min
    elif func_type is 'quantile': return n_quantile
    elif func_type is 'variance': return neighbourhood_n_deviations
    elif func_type is 'skew': return n_skew
    elif func_type is 'kurtosis': return n_kurtosis
    elif func_type is 'mad': return n_mad
    elif func_type is 'mode': return n_mode
    elif func_type is 'signal_to_noise': return n_snr
    
    raise Exception('Unable to find filter type!')


cdef void loop(
    double [:, :, ::1] arr,
    double [:, :, ::1] kernel,
    double [:, ::1] result,
    int x_max,
    int y_max,
    int z_max,
    int x_kernel_size,
    int y_kernel_size,
    int z_kernel_size,
    int non_zero,
    bint has_nodata,
    double fill_value,
    f_type apply,
) nogil:
    cdef int x, y, n, z, ni, offset_x, offset_y, offset_z
    cdef Neighbourhood * neighbourhood
    cdef bint value_is_nodata
    
    cdef int x_max_adj = x_max - 1
    cdef int y_max_adj = y_max - 1
    cdef int z_max_adj = z_max - 1

    cdef int x_edge_low = x_kernel_size
    cdef int y_edge_low = y_kernel_size
    cdef int z_edge_low = z_kernel_size

    cdef int x_edge_high = x_max_adj - x_kernel_size
    cdef int y_edge_high = y_max_adj - y_kernel_size
    cdef int z_edge_high = z_max_adj - z_kernel_size

    cdef int neighbourhood_size = sizeof(Neighbourhood) * (non_zero * z_max)

    cdef Offset * offsets = generate_offsets(kernel, x_kernel_size, y_kernel_size, z_kernel_size, non_zero)

    for x in prange(x_max):
        for y in range(y_max):

            neighbourhood = <Neighbourhood*> malloc(neighbourhood_size)
            value_is_nodata = False

            for z in range(z_max):

                if has_nodata is True and arr[x][y][z] == fill_value:
                    value_is_nodata = True
                    continue

                for n in range(non_zero):
                    ni = n * (z + 1)

                    offset_x = x + offsets[n].x
                    offset_y = y + offsets[n].y
                    offset_z = z + offsets[n].z

                    if offset_x < 0:
                        offset_x = 0
                    elif offset_x > x_max_adj:
                        offset_x = x_max_adj

                    if offset_y < 0:
                        offset_y = 0
                    elif offset_y > y_max_adj:
                        offset_y = y_max_adj

                    if offset_z < 0:
                        offset_z = 0
                    elif offset_z > z_max_adj:
                        offset_z = z_max_adj

                    neighbourhood[ni].value = arr[offset_x][offset_y][offset_z]
                    neighbourhood[ni].weight = offsets[n].weight / z_max

                    if has_nodata is True and neighbourhood[ni].value == fill_value:
                        neighbourhood[ni].weight = 0
                    else:    
                        neighbourhood[ni].weight = offsets[n].weight / z_max
            
            if has_nodata is True or (x < x_edge_low or x > x_edge_high or y < y_edge_low or y > y_edge_high or z < z_edge_low or z > z_edge_high):
                normalise_neighbourhood(neighbourhood, non_zero)

            if value_is_nodata is False:
                result[x][y][z] = apply(neighbourhood, non_zero)
            else:
                result[x][y][z] = fill_value

            free(neighbourhood)


def kernel_filter(arr, kernel, str func_type):
    offsets_py, weights_py = generate_offsets(kernel)

    cdef int kernel_size = kernel.size
    cdef int offsets [:, ::1] = offsets_py
    cdef float weights [::1] = weights_py
    cdef int arr_shape [::1] = np.array([arr.shape[0], arr.shape[1], arr.shape[2]], dtype=int)

    cdef bint has_nodata = False
    cdef float fill_value
    cdef f_type apply = func_selector(func_type)

    arr = arr.astype(np.float) if arr.dtype != np.float else arr
    result = np.empty((arr.shape[1], arr.shape[2]), dtype=np.flaot)

    cdef float [:, :, ::1] arr
    cdef float [:, ::1] result_view = result

    arr_mask = False
    if isinstance(arr, np.ma.MaskedArray):
        fill_value = arr.fill_value
        has_nodata = 1
        if arr.mask is not False:
            arr_mask = np.ma.getmask(arr)
        arr = arr.filled()
    else:
        fill_value = 0.0

    loop(
        arr,
        offsets,
        weights,
        result_view,
        arr_shape,
        kernel_size
        has_nodata,
        fill_value,
        apply,
    )

    if has_nodata is True:
        if arr_mask is not False:
            return np.ma.masked_where(arr_mask, result)
        else:
            return np.ma.masked_equal(result, fill_value)

    return result.astype(arr.dtype)
