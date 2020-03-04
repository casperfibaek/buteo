# distutils: language=py3
# cython: boundscheck=False, wraparound=False
cimport cython
# cimport numpy as cnp
from cython.parallel import prange
from cython.view cimport array as cvarray
from libc.stdlib cimport malloc, free
import numpy as np


cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compar)(const_void *, const_void *)) nogil

cdef extern from "math.h":
    double sqrt(double m)

cdef int count_non_zero(double[:, ::1] kernel, int width) nogil:
    cdef int x, y, accum

    accum = 0
    for x in prange(width):
        for y in range(width):
            if kernel[x, y] != 0:
                accum += 1

    return accum


cdef double sum_arr(double[:, ::1] kernel, int width) nogil:
    cdef int x, y
    cdef double accum

    accum = 0
    for x in prange(width):
        for y in range(width):
            if kernel[x, y] != 0:
                accum += kernel[x, y]

    return accum


cdef double* generate_weights(double [:, ::1] kernel, int width, int number_of_nonzero) nogil:
    cdef double *offsets_weights = <double*> malloc(sizeof(double) * number_of_nonzero)

    cdef int step = 0
    cdef int x, y
    for x in range(width):
        for y in range(width):
            if kernel[x, y] != 0:
                offsets_weights[step] = kernel[x][y]
                step += 1
    
    return offsets_weights


cdef int [:, ::1] generate_offsets(double [:, ::1] kernel, int width, int number_of_nonzero) nogil:
    cdef int[:, ::1] offsets
    
    with gil:
        offsets = cvarray(shape=(number_of_nonzero, 2), itemsize=sizeof(int), format="i")

    cdef int radius = <int>(width / 2)

    cdef int step = 0
    cdef int x, y
    for x in range(width):
        for y in range(width):
            if kernel[x, y] != 0.0:
                offsets[step, 0] = x - radius
                offsets[step, 1] = y - radius
                step += 1
    
    return offsets


cdef struct IndexedElement:
    int index
    double value


cdef int _compare(const_void *a, const_void *b) nogil:
    cdef double v = (<IndexedElement*> a).value-(<IndexedElement*> b).value
    if v < 0: return -1
    if v >= 0: return 1


cdef void argsort(double* data, int* order, int length) nogil:
    cdef int i
    cdef int n = length
    
    # Allocate index tracking array.
    cdef IndexedElement *order_struct = <IndexedElement *> malloc(n * sizeof(IndexedElement))
    
    # Copy data into index tracking array.
    for i in range(n):
        order_struct[i].index = i
        order_struct[i].value = data[i]
        
    # Sort index tracking array.
    qsort(<void *> order_struct, n, sizeof(IndexedElement), _compare)
    
    # Copy indices from index tracking array to output array.
    for i in range(n):
        order[i] = order_struct[i].index
        
    # Free index tracking array.
    free(order_struct)

cdef double quintile(double* arr, double* weights, int length, double quint) nogil:
    cdef int i, j, q, p, ia

    cdef double med, key, merge, sort_weight_half, top, bot, s,

    cdef double half = 2.0
    cdef double weight_sum = 0
    cdef double cumsum = 0

    cdef double* sort_arr = <double*> malloc(sizeof(double) * length)
    cdef double* sort_weight = <double*> malloc(sizeof(double) * length)
    cdef double* weighted_quantile = <double*> malloc(sizeof(double) * length)
    cdef int* order = <int*> malloc(sizeof(int) * length)

    # There is a memory usage issue with argsort. Balloons the memory.
    argsort(arr, order, length)
    
    for i in range(length):
        sort_arr[i] = arr[order[i]]
        sort_weight[i] = weights[order[i]]
        weight_sum += weights[i]

    for i in range(length):
        cumsum += sort_weight[i]
        weighted_quantile[i] = (cumsum - (quint * sort_weight[i])) / weight_sum

    med = sort_arr[length - 1]
    for i in range(length):
        if weighted_quantile[i] >= quint:
            if i == 0 or weighted_quantile[i] == quint:
                med = sort_arr[i]
                break
            top = weighted_quantile[i] - quint
            bot = quint - weighted_quantile[i - 1]
            s = top + bot
            if s is 0:
                top = 1
                bot = 1
            else:
                top = 1 - (top / s)
                bot = 1 - (bot / s)

            med = (sort_arr[i - 1] * bot) + (sort_arr[i] * top)
            break

    free(sort_arr)
    free(sort_weight)
    free(order)
    free(weighted_quantile)

    return med

cdef double median(double* arr, double* weights, int length) nogil:
    cdef int i, j, q, p, ia

    cdef double med, key, merge, sort_weight_half, top, bot, s,

    cdef double half = 2.0
    cdef double weight_sum = 0
    cdef double cumsum = 0

    cdef double* sort_arr = <double*> malloc(sizeof(double) * length)
    cdef double* sort_weight = <double*> malloc(sizeof(double) * length)
    cdef double* weighted_quantile = <double*> malloc(sizeof(double) * length)
    cdef int* order = <int*> malloc(sizeof(int) * length)

    # There is a memory usage issue with argsort. Balloons the memory.
    argsort(arr, order, length)
    
    for i in range(length):
        sort_arr[i] = arr[order[i]]
        sort_weight[i] = weights[order[i]]
        weight_sum += weights[i]

    for i in range(length):
        cumsum += sort_weight[i]
        weighted_quantile[i] = (cumsum - (0.5 * sort_weight[i])) / weight_sum

    med = sort_arr[length - 1]
    for i in range(length):
        if weighted_quantile[i] >= 0.5:
            if i == 0 or weighted_quantile[i] == 0.5:
                med = sort_arr[i]
                break
            top = weighted_quantile[i] - 0.5
            bot = 0.5 - weighted_quantile[i - 1]
            s = top + bot
            top = 1 - (top / s)
            bot = 1 - (bot / s)

            med = (sort_arr[i - 1] * bot) + (sort_arr[i] * top)
            break

    free(sort_arr)
    free(sort_weight)
    free(order)
    free(weighted_quantile)

    return med


cdef double mad(double* arr, double* weights, int length,double ptn) nogil:
    cdef int i, j, q, p

    cdef double med, mad, key, merge, sort_weight_half, top, bot, s, w

    cdef double half = 2.0
    cdef double weight_sum = 0
    cdef double cumsum = 0

    cdef double* sort_arr = <double*> malloc(sizeof(double) * length)
    cdef double* deviations = <double*> malloc(sizeof(double) * length)
    cdef double* sort_weight = <double*> malloc(sizeof(double) * length)
    cdef double* weighted_quantile = <double*> malloc(sizeof(double) * length)

    for i in range(length):
        sort_arr[i] = arr[i]
        sort_weight[i] = weights[i]
        weight_sum += weights[i]

    for i in range(1, length): 
        key = sort_arr[i]
        key_w = sort_weight[i]

        j = i - 1

        while j >= 0 and key < sort_arr[j]:
            q = j + 1
            sort_arr[q] = sort_arr[j]
            sort_weight[q] = sort_weight[j]
            j -= 1
        
        p = j + 1
        sort_arr[p] = key
        sort_weight[p] = key_w

    for i in range(length):
        sort_weight_half = 0.5 * sort_weight[i]
        cumsum += sort_weight[i]

        weighted_quantile[i] = (cumsum - sort_weight_half) / weight_sum

    med = sort_arr[length - 1]
    for i in range(length):
        if weighted_quantile[i] >= 0.5:
            if i == 0 or weighted_quantile[i] == 0.5:
                med = sort_arr[i]
                break
            top = weighted_quantile[i] - 0.5
            bot = 0.5 - weighted_quantile[i - 1]
            s = top + bot
            top = 1 - (top / s)
            bot = 1 - (bot / s)

            med = (sort_arr[i - 1] * bot) + (sort_arr[i] * top)
            break

    ########## MAD MAD MAD ###########
    
    for i in range(length):
        w = (med - sort_arr[i])

        if w < 0:
            w = w * -1.0

        deviations[i] = w

    for i in range(1, length): 
        key = deviations[i]
        key_w = sort_weight[i]

        j = i - 1

        while j >= 0 and key < deviations[j]:
            q = j + 1
            deviations[q] = deviations[j]
            sort_weight[q] = sort_weight[j]
            j -= 1
        
        p = j + 1
        deviations[p] = key
        sort_weight[p] = key_w

    cumsum = 0

    for i in range(length):
        sort_weight_half = 0.5 * sort_weight[i]
        cumsum += sort_weight[i]

        weighted_quantile[i] = (cumsum - sort_weight_half) / weight_sum

    mad = deviations[length - 1]
    for i in range(length):
        if weighted_quantile[i] >= 0.5:
            if i == 0 or weighted_quantile[i] == 0.5:
                mad = deviations[i]
                break
            top = weighted_quantile[i] - 0.5
            bot = 0.5 - weighted_quantile[i - 1]
            s = top + bot
            top = 1 - (top / s)
            bot = 1 - (bot / s)

            mad = (deviations[i - 1] * bot) + (deviations[i] * top)
            break

    ######## MAD MAD MAD #############

    free(deviations)
    free(sort_arr)
    free(sort_weight)
    free(weighted_quantile)

    # return mad
    return mad


def filter_2d(double [:, ::1] arr, double [:, ::1] kernel):
    cdef int x_max = arr.shape[0]
    cdef int y_max = arr.shape[1]
    cdef int x_max_adj = arr.shape[0] - 1
    cdef int y_max_adj = arr.shape[1] - 1
    cdef int width = kernel.shape[0]

    cdef double[:, ::1] kernel_view = kernel
    cdef int number_of_nonzero = count_non_zero(kernel_view, width)
    
    cdef int[:, ::1] offsets = generate_offsets(kernel_view, width, number_of_nonzero)
    cdef double* weights = generate_weights(kernel_view, width, number_of_nonzero)

    cdef double[:, ::1] arr_view = arr

    result = np.zeros((x_max, y_max), dtype=np.double)
    cdef double[:, ::1] result_view = result

    cdef int x, y, n, offset_x, offset_y
    for x in prange(x_max, nogil=True):
        for y in range(y_max):
            for n in range(number_of_nonzero):
                offset_x = x + offsets[n, 0]
                offset_y = y + offsets[n, 1]

                if offset_x < 0:
                    offset_x = 0
                elif offset_x > x_max_adj:
                    offset_x = x_max_adj

                if offset_y < 0:
                    offset_y = 0
                elif offset_y > y_max_adj:
                    offset_y = y_max_adj

                result_view[x][y] += arr_view[offset_x, offset_y] * weights[n]

    return result

def filter_stdev(double [:, ::1] arr, double [:, ::1] kernel):
    cdef int x_max = arr.shape[0]
    cdef int y_max = arr.shape[1]
    cdef int x_max_adj = arr.shape[0] - 1
    cdef int y_max_adj = arr.shape[1] - 1
    cdef int width = kernel.shape[0]

    cdef double[:, ::1] kernel_view = kernel
    cdef int number_of_nonzero = count_non_zero(kernel_view, width)
    cdef double sum_of_weights = sum_arr(kernel_view, width)
    
    cdef int[:, ::1] offsets = generate_offsets(kernel_view, width, number_of_nonzero)
    cdef double* weights = generate_weights(kernel_view, width, number_of_nonzero)
    cdef double non_zero_stdev = (<double> number_of_nonzero - 1.0) / (<double> number_of_nonzero)

    cdef double[:, ::1] arr_view = arr

    result = np.zeros((x_max, y_max), dtype=np.double)
    cdef double[:, ::1] result_view = result

    cdef int x, y, n, m, offset_x, offset_y
    cdef double weighted_avg
    cdef double weighted_deviations
    for x in range(x_max):
        for y in range(y_max):

            weighted_avg = 0
            weighted_deviations = 0

            for n in range(number_of_nonzero):
                offset_x = x + offsets[n, 0]
                offset_y = y + offsets[n, 1]

                if offset_x < 0:
                    offset_x = 0
                elif offset_x > x_max_adj:
                    offset_x = x_max_adj

                if offset_y < 0:
                    offset_y = 0
                elif offset_y > y_max_adj:
                    offset_y = y_max_adj

                weighted_avg += arr_view[offset_x, offset_y] * weights[n]

            for m in range(number_of_nonzero):
                offset_x = x + offsets[m, 0]
                offset_y = y + offsets[m, 1]

                if offset_x < 0:
                    offset_x = 0
                elif offset_x > x_max_adj:
                    offset_x = x_max_adj

                if offset_y < 0:
                    offset_y = 0
                elif offset_y > y_max_adj:
                    offset_y = y_max_adj

                weighted_deviations += weights[m] * ((arr_view[offset_x, offset_y] - weighted_avg) ** 2)

            result_view[x][y] = sqrt(weighted_deviations / ((non_zero_stdev - 1) * sum_of_weights) / non_zero_stdev)
            

    return result


def filter_skew_fp(double [:, ::1] arr, double [:, ::1] kernel):
    cdef int x_max = arr.shape[0]
    cdef int y_max = arr.shape[1]
    cdef int x_max_adj = arr.shape[0] - 1
    cdef int y_max_adj = arr.shape[1] - 1
    cdef int width = kernel.shape[0]

    cdef double[:, ::1] kernel_view = kernel
    cdef int number_of_nonzero = count_non_zero(kernel_view, width)
    cdef double sum_of_weights = sum_arr(kernel_view, width)
    
    cdef int[:, ::1] offsets = generate_offsets(kernel_view, width, number_of_nonzero)
    cdef double* weights = generate_weights(kernel_view, width, number_of_nonzero)
    cdef double non_zero_stdev = (<double> number_of_nonzero - 1.0) / (<double> number_of_nonzero)

    cdef double[:, ::1] arr_view = arr

    result = np.zeros((x_max, y_max), dtype=np.double)
    cdef double[:, ::1] result_view = result

    cdef int x, y, n, m, offset_x, offset_y
    cdef double weighted_avg, weighted_stdev, weighted_deviations, weighted_deviations_3
    # cdef double correction
    for x in range(x_max):
        for y in range(y_max):

            weighted_stdev = 0
            weighted_avg = 0
            weighted_deviations = 0
            weighted_deviations_3 = 0

            for n in range(number_of_nonzero):
                offset_x = x + offsets[n, 0]
                offset_y = y + offsets[n, 1]

                if offset_x < 0:
                    offset_x = 0
                elif offset_x > x_max_adj:
                    offset_x = x_max_adj

                if offset_y < 0:
                    offset_y = 0
                elif offset_y > y_max_adj:
                    offset_y = y_max_adj

                weighted_avg += arr_view[offset_x, offset_y] * weights[n]

            for m in range(number_of_nonzero):
                offset_x = x + offsets[m, 0]
                offset_y = y + offsets[m, 1]

                if offset_x < 0:
                    offset_x = 0
                elif offset_x > x_max_adj:
                    offset_x = x_max_adj

                if offset_y < 0:
                    offset_y = 0
                elif offset_y > y_max_adj:
                    offset_y = y_max_adj

                weighted_deviations += weights[m] * ((arr_view[offset_x, offset_y] - weighted_avg) ** 2)
                weighted_deviations_3 += weights[m] * ((arr_view[offset_x, offset_y] - weighted_avg) ** 3)

            weighted_stdev = sqrt(weighted_deviations / ((non_zero_stdev - 1) * sum_of_weights) / non_zero_stdev)
            # correction = sqrt(<double>number_of_nonzero * (<double>number_of_nonzero - 1)) / (<double>number_of_nonzero - 2)
            top = weighted_deviations_3 / sum_of_weights
            bot = weighted_stdev ** 3

            # result_view[x][y] = correction * (top / bot)
            result_view[x][y] = top / bot
            

    return result

def filter_kurtosis_excess(double [:, ::1] arr, double [:, ::1] kernel):
    cdef int x_max = arr.shape[0]
    cdef int y_max = arr.shape[1]
    cdef int x_max_adj = arr.shape[0] - 1
    cdef int y_max_adj = arr.shape[1] - 1
    cdef int width = kernel.shape[0]

    cdef double[:, ::1] kernel_view = kernel
    cdef int number_of_nonzero = count_non_zero(kernel_view, width)
    cdef double sum_of_weights = sum_arr(kernel_view, width)
    
    cdef int[:, ::1] offsets = generate_offsets(kernel_view, width, number_of_nonzero)
    cdef double* weights = generate_weights(kernel_view, width, number_of_nonzero)
    cdef double non_zero_stdev = (<double> number_of_nonzero - 1.0) / (<double> number_of_nonzero)

    cdef double[:, ::1] arr_view = arr

    result = np.zeros((x_max, y_max), dtype=np.double)
    cdef double[:, ::1] result_view = result

    cdef int x, y, n, m, offset_x, offset_y
    cdef double weighted_avg, weighted_stdev, weighted_deviations, weighted_deviations_3, correction
    for x in range(x_max):
        for y in range(y_max):

            weighted_stdev = 0
            weighted_avg = 0
            weighted_deviations = 0
            weighted_deviations_3 = 0

            for n in range(number_of_nonzero):
                offset_x = x + offsets[n, 0]
                offset_y = y + offsets[n, 1]

                if offset_x < 0:
                    offset_x = 0
                elif offset_x > x_max_adj:
                    offset_x = x_max_adj

                if offset_y < 0:
                    offset_y = 0
                elif offset_y > y_max_adj:
                    offset_y = y_max_adj

                weighted_avg += arr_view[offset_x, offset_y] * weights[n]

            for m in range(number_of_nonzero):
                offset_x = x + offsets[m, 0]
                offset_y = y + offsets[m, 1]

                if offset_x < 0:
                    offset_x = 0
                elif offset_x > x_max_adj:
                    offset_x = x_max_adj

                if offset_y < 0:
                    offset_y = 0
                elif offset_y > y_max_adj:
                    offset_y = y_max_adj

                weighted_deviations += weights[m] * ((arr_view[offset_x, offset_y] - weighted_avg) ** 2)
                weighted_deviations_3 += weights[m] * ((arr_view[offset_x, offset_y] - weighted_avg) ** 4)

            weighted_stdev = sqrt(weighted_deviations / (non_zero_stdev * sum_of_weights))
            top = weighted_deviations_3 / sum_of_weights
            bot = weighted_stdev ** 4

            result_view[x][y] = (top / bot) - 3
            

    return result

def filter_skew_p2(double [:, ::1] arr, double [:, ::1] kernel):
    cdef int x_max = arr.shape[0]
    cdef int y_max = arr.shape[1]
    cdef int x_max_adj = arr.shape[0] - 1
    cdef int y_max_adj = arr.shape[1] - 1
    cdef int width = kernel.shape[0]

    cdef double[:, ::1] kernel_view = kernel
    cdef int number_of_nonzero = count_non_zero(kernel_view, width)
    cdef double sum_of_weights = sum_arr(kernel_view, width)
    
    cdef int[:, ::1] offsets = generate_offsets(kernel_view, width, number_of_nonzero)
    cdef double* weights = generate_weights(kernel_view, width, number_of_nonzero)
    cdef double non_zero_stdev = (<double> number_of_nonzero - 1.0) / (<double> number_of_nonzero)

    cdef double[:, ::1] arr_view = arr

    result = np.zeros((x_max, y_max), dtype=np.double)
    cdef double[:, ::1] result_view = result

    cdef double *neighborhood
    cdef int x, y, n, m, offset_x, offset_y
    cdef double weighted_avg, weighted_stdev, weighted_deviations, weighted_deviations_3, correction
    for x in range(x_max):
        for y in range(y_max):

            weighted_stdev = 0
            weighted_avg = 0
            weighted_deviations = 0
            neighborhood = <double*> malloc(sizeof(double) * number_of_nonzero) 

            for n in range(number_of_nonzero):
                offset_x = x + offsets[n, 0]
                offset_y = y + offsets[n, 1]

                if offset_x < 0:
                    offset_x = 0
                elif offset_x > x_max_adj:
                    offset_x = x_max_adj

                if offset_y < 0:
                    offset_y = 0
                elif offset_y > y_max_adj:
                    offset_y = y_max_adj

                neighborhood[n] = arr_view[offset_x, offset_y]
                weighted_avg += arr_view[offset_x, offset_y] * weights[n]

            for m in range(number_of_nonzero):
                offset_x = x + offsets[m, 0]
                offset_y = y + offsets[m, 1]

                if offset_x < 0:
                    offset_x = 0
                elif offset_x > x_max_adj:
                    offset_x = x_max_adj

                if offset_y < 0:
                    offset_y = 0
                elif offset_y > y_max_adj:
                    offset_y = y_max_adj

                weighted_deviations += weights[m] * ((arr_view[offset_x, offset_y] - weighted_avg) ** 2)

            weighted_stdev = sqrt(weighted_deviations / ((non_zero_stdev - 1) * sum_of_weights) / non_zero_stdev)

            if weighted_stdev is 0:
                result_view[x][y] = 0
            else:
                med = median(neighborhood, weights, number_of_nonzero)
                result_view[x][y] = 3 * ((weighted_avg - med) / weighted_stdev)

    return result

def filter_skew_g(double [:, ::1] arr, double [:, ::1] kernel):
    cdef int x_max = arr.shape[0]
    cdef int y_max = arr.shape[1]
    cdef int x_max_adj = arr.shape[0] - 1
    cdef int y_max_adj = arr.shape[1] - 1
    cdef int width = kernel.shape[0]

    cdef double[:, ::1] kernel_view = kernel

    cdef int number_of_nonzero = count_non_zero(kernel_view, width)
    
    cdef int[:, ::1] offsets = generate_offsets(kernel_view, width, number_of_nonzero)
    cdef double* weights = generate_weights(kernel_view, width, number_of_nonzero)

    cdef double[:, ::1] arr_view = arr

    result = np.zeros((x_max, y_max), dtype=np.double)
    cdef double[:, ::1] result_view = result

    cdef double *neighborhood
    cdef double q1, q2, q3
    cdef int x, y, n, offset_x, offset_y
    for x in prange(x_max, nogil=True):
        for y in range(y_max):
            neighborhood = <double*> malloc(sizeof(double) * number_of_nonzero) 
            for n in range(number_of_nonzero):
                offset_x = x + offsets[n, 0]
                offset_y = y + offsets[n, 1]

                if offset_x < 0:
                    offset_x = 0
                elif offset_x > x_max_adj:
                    offset_x = x_max_adj

                if offset_y < 0:
                    offset_y = 0
                elif offset_y > y_max_adj:
                    offset_y = y_max_adj

                neighborhood[n] = arr_view[offset_x, offset_y]
            
            q1 = quintile(neighborhood, weights, number_of_nonzero, 0.25)
            q2 = median(neighborhood, weights, number_of_nonzero)
            q3 = quintile(neighborhood, weights, number_of_nonzero, 0.75)

            if (q3 - q1) == 0:
                result_view[x][y] = 0
            else:
                result_view[x][y] = (q1 + q3 - (2 * q2)) / (q3 - q1)

            free(neighborhood)

    return result

def filter_variance(double [:, ::1] arr, double [:, ::1] kernel):
    cdef int x_max = arr.shape[0]
    cdef int y_max = arr.shape[1]
    cdef int x_max_adj = arr.shape[0] - 1
    cdef int y_max_adj = arr.shape[1] - 1
    cdef int width = kernel.shape[0]

    cdef double[:, ::1] kernel_view = kernel
    cdef int number_of_nonzero = count_non_zero(kernel_view, width)
    cdef double sum_of_weights = sum_arr(kernel_view, width)
    
    cdef int[:, ::1] offsets = generate_offsets(kernel_view, width, number_of_nonzero)
    cdef double* weights = generate_weights(kernel_view, width, number_of_nonzero)
    cdef double non_zero_stdev = (<double> number_of_nonzero - 1.0) / (<double> number_of_nonzero)

    cdef double[:, ::1] arr_view = arr

    result = np.zeros((x_max, y_max), dtype=np.double)
    cdef double[:, ::1] result_view = result

    cdef int x, y, n, m, offset_x, offset_y
    cdef double weighted_avg
    cdef double weighted_deviations
    # for x in prange(x_max, nogil=True):
    for x in range(x_max):
        for y in range(y_max):

            weighted_avg = 0
            weighted_deviations = 0

            for n in range(number_of_nonzero):
                offset_x = x + offsets[n, 0]
                offset_y = y + offsets[n, 1]

                if offset_x < 0:
                    offset_x = 0
                elif offset_x > x_max_adj:
                    offset_x = x_max_adj

                if offset_y < 0:
                    offset_y = 0
                elif offset_y > y_max_adj:
                    offset_y = y_max_adj

                weighted_avg += arr_view[offset_x, offset_y] * weights[n]

            for m in range(number_of_nonzero):
                offset_x = x + offsets[m, 0]
                offset_y = y + offsets[m, 1]

                if offset_x < 0:
                    offset_x = 0
                elif offset_x > x_max_adj:
                    offset_x = x_max_adj

                if offset_y < 0:
                    offset_y = 0
                elif offset_y > y_max_adj:
                    offset_y = y_max_adj

                weighted_deviations += weights[m] * ((arr_view[offset_x, offset_y] - weighted_avg) ** 2)

            result_view[x][y] = weighted_deviations / ((non_zero_stdev - 1) * sum_of_weights) / non_zero_stdev
            

    return result

def filter_median(double [:, ::1] arr, double [:, ::1] kernel):
    cdef int x_max = arr.shape[0]
    cdef int y_max = arr.shape[1]
    cdef int x_max_adj = arr.shape[0] - 1
    cdef int y_max_adj = arr.shape[1] - 1
    cdef int width = kernel.shape[0]

    cdef double[:, ::1] kernel_view = kernel

    cdef int number_of_nonzero = count_non_zero(kernel_view, width)
    
    cdef int[:, ::1] offsets = generate_offsets(kernel_view, width, number_of_nonzero)
    cdef double* weights = generate_weights(kernel_view, width, number_of_nonzero)

    cdef double[:, ::1] arr_view = arr

    result = np.zeros((x_max, y_max), dtype=np.double)
    cdef double[:, ::1] result_view = result

    cdef double *neighborhood
    cdef int x, y, n, offset_x, offset_y
    for x in prange(x_max, nogil=True):
        for y in range(y_max):
            neighborhood = <double*> malloc(sizeof(double) * number_of_nonzero) 
            for n in range(number_of_nonzero):
                offset_x = x + offsets[n, 0]
                offset_y = y + offsets[n, 1]

                if offset_x < 0:
                    offset_x = 0
                elif offset_x > x_max_adj:
                    offset_x = x_max_adj

                if offset_y < 0:
                    offset_y = 0
                elif offset_y > y_max_adj:
                    offset_y = y_max_adj

                neighborhood[n] = arr_view[offset_x, offset_y]
            
            result_view[x][y] = median(neighborhood, weights, number_of_nonzero)
            free(neighborhood)

    return result

def filter_mad(double [:, ::1] arr, double [:, ::1] kernel):
    cdef int x_max = arr.shape[0]
    cdef int y_max = arr.shape[1]
    cdef int x_max_adj = arr.shape[0] - 1
    cdef int y_max_adj = arr.shape[1] - 1
    cdef int width = kernel.shape[0]

    cdef double[:, ::1] kernel_view = kernel

    cdef int number_of_nonzero = count_non_zero(kernel_view, width)
    
    cdef int[:, ::1] offsets = generate_offsets(kernel_view, width, number_of_nonzero)
    cdef double* weights = generate_weights(kernel_view, width, number_of_nonzero)

    cdef double[:, ::1] arr_view = arr

    result = np.zeros((x_max, y_max), dtype=np.double)
    cdef double[:, ::1] result_view = result

    cdef double *neighborhood
    cdef int x, y, n, offset_x, offset_y
    for x in prange(x_max, nogil=True):
        for y in range(y_max):
            neighborhood = <double*> malloc(sizeof(double) * number_of_nonzero) 
            for n in range(number_of_nonzero):
                offset_x = x + offsets[n, 0]
                offset_y = y + offsets[n, 1]

                if offset_x < 0:
                    offset_x = 0
                elif offset_x > x_max_adj:
                    offset_x = x_max_adj

                if offset_y < 0:
                    offset_y = 0
                elif offset_y > y_max_adj:
                    offset_y = y_max_adj

                neighborhood[n] = arr_view[offset_x, offset_y]

            result_view[x][y] = mad(neighborhood, weights, number_of_nonzero, arr_view[x, y])
            free(neighborhood)

    return result

# python setup.py build_ext --inplace