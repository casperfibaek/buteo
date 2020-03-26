# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, profile = False
cimport cython
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, pow, fabs
import numpy as np


cdef extern from "<float.h>":
    const double DBL_MAX
    const double DBL_MIN


cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size, int(*compar)(const_void *, const_void *)) nogil


cdef struct Neighbourhood:
    double value
    double weight


cdef struct Offset:
  int x
  int y
  double weight


ctypedef double (*f_type) (Neighbourhood *, int) nogil


cdef int compare(const_void *a, const_void *b) nogil:
    cdef double v = (<Neighbourhood*> a).value - (<Neighbourhood*> b).value
    if v < 0: return -1
    if v > 0: return 1
    return 0

cdef double neighbourhood_sum(Neighbourhood * neighbourhood, int non_zero) nogil:
    cdef int x
    cdef double accum

    accum = 0
    for x in range(non_zero):
        accum += neighbourhood[x].value * neighbourhood[x].weight

    return accum


cdef double neighbourhood_weight_sum(Neighbourhood * neighbourhood, int non_zero) nogil:
    cdef int x
    cdef double accum

    accum = 0
    for x in range(non_zero):
        accum += neighbourhood[x].weight

    return accum


cdef void normalise_neighbourhood(Neighbourhood * neighbourhood, int non_zero) nogil:
    cdef int x
    cdef double sum_of_weights = neighbourhood_weight_sum(neighbourhood, non_zero)

    for x in range(non_zero):
        neighbourhood[x].weight = neighbourhood[x].weight /sum_of_weights


cdef double neighbourhood_weighted_quintile(Neighbourhood * neighbourhood, int non_zero, double quintile) nogil:
    cdef double weighted_quantile, top, bot, tb
    cdef int i

    cdef double sum_of_weights = neighbourhood_weight_sum(neighbourhood, non_zero)
    cdef double cumsum = 0
    cdef double* w_cum = <double*> malloc(sizeof(double) * non_zero)

    qsort(<void *> neighbourhood, non_zero, sizeof(Neighbourhood), compare)
  
    weighted_quantile = neighbourhood[non_zero - 1].value
    for i in range(non_zero):
        cumsum += neighbourhood[i].weight
        w_cum[i] = (cumsum - (quintile * neighbourhood[i].weight)) / sum_of_weights

        if cumsum >= quintile:
            if i == 0 or w_cum[i] == quintile:
                weighted_quantile = neighbourhood[i].value
                break
                
            top = w_cum[i] - quintile
            bot = quintile - w_cum[i - 1]
            tb = top + bot

            top = 1 - (top / tb)
            bot = 1 - (bot / tb)

            weighted_quantile = (neighbourhood[i - 1].value * bot) + (neighbourhood[i].value * top)
            break

    free(w_cum)
    return weighted_quantile


cdef double neighbourhood_max(Neighbourhood * neighbourhood, int non_zero) nogil:
    cdef int x, y, current_max_i
    cdef double current_max, val

    current_max = neighbourhood[0].value * neighbourhood[0].weight
    current_max_i = 0
    for x in range(non_zero):
        val = neighbourhood[x].value * neighbourhood[x].weight
        if val > current_max:
            current_max = val
            current_max_i = x

    return neighbourhood[current_max_i].value * neighbourhood[current_max_i].weight


cdef double neighbourhood_min(Neighbourhood * neighbourhood, int non_zero) nogil:
    cdef int x, y, current_min_i
    cdef double current_min

    current_min = DBL_MAX

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


cdef double weighted_variance(Neighbourhood * neighbourhood, int non_zero, int power) nogil:
    cdef int x, y
    cdef double accum, weighted_average, deviations
    cdef double sum_of_weights = neighbourhood_weight_sum(neighbourhood, non_zero)

    weighted_average = neighbourhood_sum(neighbourhood, non_zero)

    deviations = 0
    for x in range(non_zero):
        deviations += neighbourhood[x].weight * (pow((neighbourhood[x].value - weighted_average), power))

    return deviations / sum_of_weights


cdef double neighbourhood_weighted_variance(Neighbourhood * neighbourhood, int non_zero) nogil:
    cdef double variance = weighted_variance(neighbourhood, non_zero, 2)
    return variance


cdef double neighbourhood_weighted_standard_deviation(Neighbourhood * neighbourhood, int non_zero) nogil:
    cdef double variance = weighted_variance(neighbourhood, non_zero, 2)
    cdef double standard_deviation = sqrt(variance)

    return standard_deviation


cdef double neighbourhood_weighted_q1(Neighbourhood * neighbourhood, int non_zero) nogil:
    cdef double weighted_q1 = neighbourhood_weighted_quintile(neighbourhood, non_zero, 0.25)
    return weighted_q1


cdef double neighbourhood_weighted_median(Neighbourhood * neighbourhood, int non_zero) nogil:
    cdef double weighted_median = neighbourhood_weighted_quintile(neighbourhood, non_zero, 0.5)
    return weighted_median


cdef double neighbourhood_weighted_q3(Neighbourhood * neighbourhood, int non_zero) nogil:
    cdef double weighted_q3 = neighbourhood_weighted_quintile(neighbourhood, non_zero, 0.75)
    return weighted_q3


cdef double neighbourhood_weighted_mad(Neighbourhood * neighbourhood, int non_zero) nogil:
    cdef double weighted_median = neighbourhood_weighted_median(neighbourhood, non_zero)
    cdef Neighbourhood * deviations = <Neighbourhood*> malloc(sizeof(Neighbourhood) * non_zero)

    for x in range(non_zero):
        deviations[x].value = fabs(neighbourhood[x].value - weighted_median)
        deviations[x].weight = neighbourhood[x].weight

    cdef double mad = neighbourhood_weighted_median(deviations, non_zero)

    free(deviations)

    return mad


cdef double neighbourhood_weighted_mad_std(Neighbourhood * neighbourhood, int non_zero) nogil:
    cdef double mad_std = neighbourhood_weighted_mad(neighbourhood, non_zero) * 1.4826
    return mad_std


cdef double neighbourhood_weighted_skew_fp(Neighbourhood * neighbourhood, int non_zero) nogil:
    cdef double standard_deviation = neighbourhood_weighted_standard_deviation(neighbourhood, non_zero)

    if standard_deviation == 0:
        return 0

    cdef double variance_3 = weighted_variance(neighbourhood, non_zero, 3)

    return variance_3 / (pow(standard_deviation, 3))


cdef double neighbourhood_weighted_skew_p2(Neighbourhood * neighbourhood, int non_zero) nogil:
    cdef double standard_deviation = neighbourhood_weighted_standard_deviation(neighbourhood, non_zero)

    if standard_deviation == 0:
        return 0

    cdef double median = neighbourhood_weighted_median(neighbourhood, non_zero)
    cdef double mean = neighbourhood_sum(neighbourhood, non_zero)

    return 3 * ((mean - median) / standard_deviation)


cdef double neighbourhood_weighted_skew_g(Neighbourhood * neighbourhood, int non_zero) nogil:
    cdef double q1 = neighbourhood_weighted_quintile(neighbourhood, non_zero, 0.25)
    cdef double q2 = neighbourhood_weighted_quintile(neighbourhood, non_zero, 0.50)
    cdef double q3 = neighbourhood_weighted_quintile(neighbourhood, non_zero, 0.75)

    cdef double iqr = q3 - q1

    if iqr == 0:
        return 0

    return (q1 + q3 - (2 * q2)) / iqr


cdef double neighbourhood_weighted_iqr(Neighbourhood * neighbourhood, int non_zero) nogil:
    cdef double q1 = neighbourhood_weighted_quintile(neighbourhood, non_zero, 0.25)
    cdef double q3 = neighbourhood_weighted_quintile(neighbourhood, non_zero, 0.75)

    cdef double iqr = q3 - q1

    return iqr


cdef double neighbourhood_weighted_kurtosis_excess(Neighbourhood * neighbourhood, int non_zero) nogil:
    cdef double standard_deviation = neighbourhood_weighted_standard_deviation(neighbourhood, non_zero)

    if standard_deviation == 0:
        return 0

    cdef double variance_4 = weighted_variance(neighbourhood, non_zero, 4)

    return (variance_4 / (pow(standard_deviation, 4))) - 3


cdef Offset * generate_offsets(double [:, ::1] kernel, int kernel_width, int non_zero) nogil:
    cdef int x, y
    cdef int radius = <int>(kernel_width / 2)
    cdef int step = 0

    cdef Offset *offsets = <Offset *> malloc(non_zero * sizeof(Offset))

    for x in range(kernel_width):
        for y in range(kernel_width):
            if kernel[x, y] != 0.0:
                offsets[step].x = x - radius
                offsets[step].y = y - radius
                offsets[step].weight = kernel[x, y]
                step += 1

    return offsets


cdef f_type func_selector(str func_type):
    if func_type is 'mean': return neighbourhood_sum
    elif func_type is 'avg': return neighbourhood_sum
    elif func_type is 'average': return neighbourhood_sum
    elif func_type is 'dilate': return neighbourhood_max
    elif func_type is 'erode': return neighbourhood_min
    elif func_type is 'median': return neighbourhood_weighted_median
    elif func_type is 'med': return neighbourhood_weighted_median
    elif func_type is 'variance': return neighbourhood_weighted_variance
    elif func_type is 'var': return neighbourhood_weighted_variance
    elif func_type is 'standard_deviation': return neighbourhood_weighted_standard_deviation
    elif func_type is 'stdev': return neighbourhood_weighted_standard_deviation
    elif func_type is 'std': return neighbourhood_weighted_standard_deviation
    elif func_type is 'q1': return neighbourhood_weighted_q1
    elif func_type is 'q3': return neighbourhood_weighted_q3
    elif func_type is 'iqr': return neighbourhood_weighted_iqr
    elif func_type is 'skew_fp': return neighbourhood_weighted_skew_fp
    elif func_type is 'skew_p2': return neighbourhood_weighted_skew_p2
    elif func_type is 'skew_g': return neighbourhood_weighted_skew_g
    elif func_type is 'kurtosis': return neighbourhood_weighted_kurtosis_excess
    elif func_type is 'mad': return neighbourhood_weighted_mad
    elif func_type is 'mad_std': return neighbourhood_weighted_mad_std
    
    raise Exception('Unable to find filter type!')


cdef void fast_2d_sum(
    double [:, ::1] arr,
    double [:, ::1] kernel,
    double [:, ::1] result,
    int x_max,
    int y_max,
    int kernel_width,
    int non_zero,
) nogil:
    cdef int x, y, n, i, j, offset_x, offset_y
    cdef int x_max_adj = x_max - 1
    cdef int y_max_adj = y_max - 1
    cdef int neighbourhood_size = sizeof(double) * non_zero
    cdef int radius = <int>(kernel_width / 2)

    cdef int step = 0
    cdef int * x_offsets = <int *> malloc(sizeof(int) * non_zero)
    cdef int * y_offsets = <int *> malloc(sizeof(int) * non_zero) 
    cdef double * weights = <double *> malloc(sizeof(double) * non_zero) 

    for i in range(kernel_width):
        for j in range(kernel_width):
            if kernel[i, j] != 0.0:
                x_offsets[step] = i - radius
                y_offsets[step] = j - radius
                weights[step] = kernel[i, j]
                step += 1

    for x in prange(x_max):
        for y in range(y_max):
            for n in range(non_zero):
                offset_x = x + x_offsets[n]
                offset_y = y + y_offsets[n]

                if offset_x < 0:
                    offset_x = 0
                elif offset_x > x_max_adj:
                    offset_x = x_max_adj
                if offset_y < 0:
                    offset_y = 0
                elif offset_y > y_max_adj:
                    offset_y = y_max_adj

                result[x][y] += arr[offset_x][offset_y] * weights[n]

    free(x_offsets)
    free(y_offsets)
    free(weights)


cdef void loop(
    double [:, ::1] arr,
    double [:, ::1] kernel,
    double [:, ::1] result,
    int x_max,
    int y_max,
    int kernel_width,
    int non_zero,
    bint has_nodata,
    double fill_value,
    f_type apply,
) nogil:
    cdef int x, y, n, offset_x, offset_y
    cdef Neighbourhood * neighbourhood
    cdef int x_max_adj = x_max - 1
    cdef int y_max_adj = y_max - 1
    cdef int x_edge_low = kernel_width
    cdef int y_edge_low = kernel_width
    cdef int x_edge_high = x_max_adj - kernel_width
    cdef int y_edge_high = y_max_adj - kernel_width
    cdef int neighbourhood_size = sizeof(Neighbourhood) * non_zero

    cdef Offset * offsets = generate_offsets(kernel, kernel_width, non_zero)

    for x in prange(x_max):
        for y in range(y_max):

            if has_nodata is True and arr[x][y] == fill_value:
                result[x][y] = fill_value
                continue

            neighbourhood = <Neighbourhood*> malloc(neighbourhood_size) 

            for n in range(non_zero):
                offset_x = x + offsets[n].x
                offset_y = y + offsets[n].y

                if offset_x < 0:
                    offset_x = 0
                elif offset_x > x_max_adj:
                    offset_x = x_max_adj
                if offset_y < 0:
                    offset_y = 0
                elif offset_y > y_max_adj:
                    offset_y = y_max_adj

                neighbourhood[n].value = arr[offset_x][offset_y]

                if has_nodata is True and neighbourhood[n].value == fill_value:
                    neighbourhood[n].weight = 0
                else:    
                    neighbourhood[n].weight = offsets[n].weight

            if has_nodata is True or (x < x_edge_low or y < y_edge_low or x > x_edge_high or y > y_edge_high):
                normalise_neighbourhood(neighbourhood, non_zero)

            result[x][y] = apply(neighbourhood, non_zero)

            free(neighbourhood)


cdef void loop_3d(
    double [:, :, ::1] arr,
    double [:, ::1] kernel,
    double [:, ::1] result,
    int z_max,
    int x_max,
    int y_max,
    int kernel_width,
    int non_zero,
    bint has_nodata,
    double fill_value,
    f_type apply,
) nogil:
    cdef int x, y, n, z, ni, offset_x, offset_y
    cdef Neighbourhood * neighbourhood
    cdef bint value_is_nodata
    cdef int x_max_adj = x_max - 1
    cdef int y_max_adj = y_max - 1
    cdef int x_edge_low = kernel_width
    cdef int y_edge_low = kernel_width
    cdef int x_edge_high = x_max_adj - kernel_width
    cdef int y_edge_high = y_max_adj - kernel_width
    cdef int neighbourhood_size = sizeof(Neighbourhood) * (non_zero * z_max)

    cdef Offset * offsets = generate_offsets(kernel, kernel_width, non_zero)

    for x in prange(x_max):
        for y in range(y_max):

            neighbourhood = <Neighbourhood*> malloc(neighbourhood_size)
            value_is_nodata = False

            for z in range(z_max):

                if has_nodata is True and arr[z][x][y] == fill_value:
                    value_is_nodata = True
                    continue

                for n in range(non_zero):
                    ni = n * (z + 1)

                    offset_x = x + offsets[n].x
                    offset_y = y + offsets[n].y

                    if offset_x < 0:
                        offset_x = 0
                    elif offset_x > x_max_adj:
                        offset_x = x_max_adj
                    if offset_y < 0:
                        offset_y = 0
                    elif offset_y > y_max_adj:
                        offset_y = y_max_adj

                    neighbourhood[ni].value = arr[z][offset_x][offset_y]
                    neighbourhood[ni].weight = offsets[n].weight / z_max

                    if has_nodata is True and neighbourhood[ni].value == fill_value:
                        neighbourhood[ni].weight = 0
                    else:    
                        neighbourhood[ni].weight = offsets[n].weight / z_max
            
            if has_nodata is True or (x < x_edge_low or y < y_edge_low or x > x_edge_high or y > y_edge_high):
                normalise_neighbourhood(neighbourhood, non_zero)

            if value_is_nodata is False:
                result[x][y] = apply(neighbourhood, non_zero)
            else:
                result[x][y] = fill_value

            free(neighbourhood)


def fast_sum(arr, kernel):
    cdef int non_zero = np.count_nonzero(kernel)

    result = np.empty((arr.shape[0], arr.shape[1]), dtype=np.double)
    arr = arr.astype(np.double) if arr.dtype != np.double else arr

    cdef double [:, ::1] arr_view = arr
    cdef double [:, ::1] result_view = result
    cdef double [:, ::1] kernel_view = kernel.astype(np.double) if kernel.dtype != np.double else kernel

    fast_2d_sum(arr_view, kernel_view, result_view, arr.shape[0], arr.shape[1], kernel.shape[0], non_zero)

    return result


def kernel_filter(arr, kernel, str func_type, dtype='float32'):
    cdef bint has_nodata = False
    cdef double fill_value
    cdef f_type apply = func_selector(func_type)
    cdef int non_zero = np.count_nonzero(kernel)
    cdef double [:, ::1] kernel_view = kernel.astype(np.double) if kernel.dtype != np.double else kernel
    cdef double [:, ::1] arr_view_2d
    cdef double [:, :, ::1] arr_view_3d
    cdef int dims = len(arr.shape)

    assert(dims == 2 or dims == 3)

    arr_mask = False
    arr = arr.astype(np.double) if arr.dtype != np.double else arr
    if dims ==2:
        result = np.empty((arr.shape[0], arr.shape[1]), dtype=np.double)
    else:
        result = np.empty((arr.shape[1], arr.shape[2]), dtype=np.double)

    cdef double [:, ::1] result_view = result

    if isinstance(arr, np.ma.MaskedArray):
        fill_value = arr.fill_value
        has_nodata = 1
        if arr.mask is not False:
            arr_mask = np.ma.getmask(arr)
        arr = arr.filled()
    else:
        fill_value = 0.0

    if dims is 3:
        arr_view_3d = arr
        loop_3d(
          arr_view_3d,
          kernel_view,
          result_view,
          arr.shape[0],
          arr.shape[1],
          arr.shape[2],
          kernel.shape[0],
          non_zero,
          has_nodata,
          fill_value,
          apply,
        )
    else:
        arr_view_2d = arr
        loop(
          arr_view_2d,
          kernel_view,
          result_view,
          arr.shape[0],
          arr.shape[1],
          kernel.shape[0],
          non_zero,
          has_nodata,
          fill_value,
          apply,
        )

    if has_nodata is True:
        if arr_mask is not False:
            return np.ma.masked_where(arr_mask, result).astype(dtype)
        else:
            return np.ma.masked_equal(result, fill_value).astype(dtype)

    return result.astype(dtype)


cdef int mode(int * a, int n) nogil:
    cdef int i, j, count
    cdef int max_value = 0
    cdef int max_count = 0

    for i in range(n):
        count = 0

        for j in range(n):
            if a[j] == a[i]:
                count += 1
        
        if count > max_count:
            max_count = count
            max_value = a[i]
    
    return max_value


cdef void loop_mode(
    int [:, ::1] arr,
    double [:, ::1] kernel,
    int [:, ::1] result,
    int x_max,
    int y_max,
    int kernel_width,
    int non_zero,
) nogil:
    cdef int x, y, n, offset_x, offset_y
    cdef int * neighbourhood
    cdef int x_max_adj = x_max - 1
    cdef int y_max_adj = y_max - 1
    cdef int neighbourhood_size = sizeof(int) * non_zero

    cdef Offset * offsets = generate_offsets(kernel, kernel_width, non_zero)

    for x in prange(x_max):
        for y in range(y_max):

            neighbourhood = <int *> malloc(neighbourhood_size)

            for n in range(non_zero):
                offset_x = x + offsets[n].x
                offset_y = y + offsets[n].y

                if offset_x < 0:
                    offset_x = 0
                elif offset_x > x_max_adj:
                    offset_x = x_max_adj
                if offset_y < 0:
                    offset_y = 0
                elif offset_y > y_max_adj:
                    offset_y = y_max_adj

                neighbourhood[n] = arr[offset_x][offset_y]

            result[x][y] = mode(neighbourhood, non_zero)

            free(neighbourhood)



def mode_array(arr, kernel):
    cdef int non_zero = np.count_nonzero(kernel)
    cdef double [:, ::1] kernel_view = kernel.astype(np.double) if kernel.dtype != np.double else kernel

    result = np.zeros(arr.shape, dtype=np.intc)
    cdef int [:, ::1] result_view = result
    cdef int [:, ::1] arr_view = arr.astype(np.intc) if arr.dtype != np.intc else arr

    loop_mode(arr_view, kernel_view, result_view, arr.shape[0], arr.shape[1], kernel.shape[0], non_zero)

    return result
