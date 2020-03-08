# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, profile = False
cimport cython
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, pow, fabs
import numpy as np


cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compar)(const_void *, const_void *)) nogil

cdef struct Neighbourhood:
  double value
  double weight

cdef struct Offset:
  int x
  int y
  double weight

cdef struct IndexedElement:
    int index
    double value

ctypedef double (*f_type) (Neighbourhood *, int, double) nogil

cdef int _compare(const_void *a, const_void *b) nogil:
  cdef double v = (<IndexedElement*> a).value - (<IndexedElement*> b).value
  if v < 0: return -1
  if v >= 0: return 1

cdef void argsort(Neighbourhood * neighbourhood, int* order, int non_zero) nogil:
  cdef int i
  
  # Allocate index tracking array.
  cdef IndexedElement *order_struct = <IndexedElement *> malloc(non_zero * sizeof(IndexedElement))
  
  # Copy data into index tracking array.
  for i in range(non_zero):
      order_struct[i].index = i
      order_struct[i].value = neighbourhood[i].value
      
  # Sort index tracking array.
  qsort(<void *> order_struct, non_zero, sizeof(IndexedElement), _compare)
  
  # Copy indices from index tracking array to output array.
  for i in range(non_zero):
      order[i] = order_struct[i].index
      
  # Free index tracking array.
  free(order_struct)

cdef double neighbourhood_sum(Neighbourhood * neighbourhood, int non_zero, double sum_of_weights) nogil:
    cdef int x, y
    cdef double accum

    accum = 0
    for x in range(non_zero):
      accum += neighbourhood[x].value * neighbourhood[x].weight

    return accum

cdef double weighted_variance(Neighbourhood * neighbourhood, int non_zero, int power, double sum_of_weights) nogil:
    cdef int x, y
    cdef double accum, weighted_average, deviations

    weighted_average = neighbourhood_sum(neighbourhood, non_zero)

    deviations = 0
    for x in range(non_zero):
      deviations += neighbourhood[x].weight * (pow((neighbourhood[x].value - weighted_average), power))

    return deviations / sum_of_weights

cdef double neighbourhood_weighted_variance(Neighbourhood * neighbourhood, int non_zero, double sum_of_weights) nogil:
  cdef double variance = weighted_variance(neighbourhood, non_zero, 2, double sum_of_weights)
  return variance

cdef double neighbourhood_weighted_standard_deviation(Neighbourhood * neighbourhood, int non_zero, double sum_of_weights) nogil:
  cdef double variance = weighted_variance(neighbourhood, non_zero, 2, double sum_of_weights)
  cdef double standard_deviation = sqrt(variance)

  return standard_deviation

cdef double neighbourhood_weighted_quintile(Neighbourhood * neighbourhood, int non_zero, double quintile, double sum_of_weights) nogil:
    cdef int i
    cdef double weighted_quantile, top, bot, s,

    cdef double cumsum = 0

    cdef double* sort_arr = <double*> malloc(sizeof(double) * non_zero)
    cdef double* sort_weight = <double*> malloc(sizeof(double) * non_zero)
    cdef double* cx = <double*> malloc(sizeof(double) * non_zero)
    cdef int* order = <int*> malloc(sizeof(int) * non_zero)

    argsort(neighbourhood, order, non_zero)
    
    for i in range(non_zero):
        cumsum += neighbourhood[order[i]].weight
        cx[i] = (cumsum - (quintile * neighbourhood[order[i]].weight)) / sum_of_weights

    for i in range(non_zero):
        if cx[i] >= quintile:
            if i == 0 or cx[i] == quintile:
                return neighbourhood[order[i]].value
            high = cx[i][v] - quintile
            low = quintile - cx[i - 1]
            weight = low / (low + high)    

            return (neighbourhood[order[i - 1]].value * weight) + (neighbourhood[order[i]].value * (1 - weight))

    free(sort_arr)
    free(sort_weight)
    free(order)
    free(cx)

    return neighbourhood[order[non_zero - 1]].value


cdef double neighbourhood_weighted_median(Neighbourhood * neighbourhood, int non_zero, double sum_of_weights) nogil:
  cdef double weighted_median = neighbourhood_weighted_quintile(neighbourhood, non_zero, 0.5, double sum_of_weights)
  return weighted_median

cdef double neighbourhood_weighted_q1(Neighbourhood * neighbourhood, int non_zero, double sum_of_weights) nogil:
  cdef double weighted_median = neighbourhood_weighted_quintile(neighbourhood, non_zero, 0.25, double sum_of_weights)
  return weighted_median

cdef double neighbourhood_weighted_q3(Neighbourhood * neighbourhood, int non_zero, double sum_of_weights) nogil:
  cdef double weighted_median = neighbourhood_weighted_quintile(neighbourhood, non_zero, 0.75, double sum_of_weights)
  return weighted_median

cdef double neighbourhood_weighted_mad(Neighbourhood * neighbourhood, int non_zero, double sum_of_weights) nogil:
  cdef double weighted_median = neighbourhood_weighted_median(neighbourhood, non_zero, double sum_of_weights)
  cdef Neighbourhood * deviations = <Neighbourhood*> malloc(sizeof(Neighbourhood) * non_zero)

  for x in range(non_zero):
    deviations[x].value = fabs(neighbourhood[x].value - weighted_median)
    deviations[x].weight = neighbourhood[x].weight

  cdef double mad = neighbourhood_weighted_median(deviations, non_zero, double sum_of_weights)

  free(deviations)

  return mad


cdef double neighbourhood_weighted_mad_std(Neighbourhood * neighbourhood, int non_zero, double sum_of_weights) nogil:
  cdef double mad_std = neighbourhood_weighted_mad(neighbourhood, non_zero, double sum_of_weights) * 1.4826
  return mad_std

cdef double neighbourhood_weighted_skew_fp(Neighbourhood * neighbourhood, int non_zero, double sum_of_weights) nogil:
  cdef double standard_deviation = neighbourhood_weighted_standard_deviation(neighbourhood, non_zero, double sum_of_weights)

  if standard_deviation == 0:
    return 0

  cdef double variance_3 = weighted_variance(neighbourhood, non_zero, 3, double sum_of_weights)
  return variance_3 / (pow(standard_deviation, 3))

cdef double neighbourhood_weighted_skew_p2(Neighbourhood * neighbourhood, int non_zero, double sum_of_weights) nogil:
  cdef double standard_deviation = neighbourhood_weighted_standard_deviation(neighbourhood, non_zero, double sum_of_weights)

  if standard_deviation == 0:
    return 0

  cdef double median = neighbourhood_weighted_median(neighbourhood, non_zero, double sum_of_weights)
  cdef double mean = neighbourhood_sum(neighbourhood, non_zero)

  return 3 * ((mean - median) / standard_deviation)

cdef double neighbourhood_weighted_skew_g(Neighbourhood * neighbourhood, int non_zero, double sum_of_weights) nogil:
  cdef double q1 = neighbourhood_weighted_quintile(neighbourhood, non_zero, 0.25, double sum_of_weights)
  cdef double q2 = neighbourhood_weighted_quintile(neighbourhood, non_zero, 0.50, double sum_of_weights)
  cdef double q3 = neighbourhood_weighted_quintile(neighbourhood, non_zero, 0.75, double sum_of_weights)

  cdef double iqr = q3 - q1

  if iqr == 0:
    return 0

  return (q1 + q3 - (2 * q2)) / iqr

cdef double neighbourhood_weighted_iqr(Neighbourhood * neighbourhood, int non_zero, double sum_of_weights) nogil:
  cdef double q1 = neighbourhood_weighted_quintile(neighbourhood, non_zero, 0.25, double sum_of_weights)
  cdef double q3 = neighbourhood_weighted_quintile(neighbourhood, non_zero, 0.75, double sum_of_weights)

  cdef double iqr = q3 - q1

  return iqr

cdef double neighbourhood_weighted_kurtosis_excess(Neighbourhood * neighbourhood, int non_zero, double sum_of_weights) nogil:
  cdef double standard_deviation = neighbourhood_weighted_standard_deviation(neighbourhood, non_zero, double sum_of_weights)

  if standard_deviation == 0:
    return 0

  cdef double variance_4 = weighted_variance(neighbourhood, non_zero, 4, double sum_of_weights)
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

cdef void loop(double [:, ::1] arr, double [:, ::1] kernel, double [:, ::1] result, int x_max, int y_max, int kernel_width, double sum_of_weights, int non_zero, f_type apply) nogil:
  cdef int x, y, n, offset_x, offset_y
  cdef Neighbourhood * neighbourhood
  
  cdef int x_max_adj = x_max - 1
  cdef int y_max_adj = y_max - 1
  cdef int neighbourhood_size = sizeof(Neighbourhood) * non_zero

  cdef Offset * offsets = generate_offsets(kernel, kernel_width, non_zero)

  for x in prange(x_max):
    for y in range(y_max):

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

        neighbourhood[n].value = arr[offset_x, offset_y]
        neighbourhood[n].weight = offsets[n].weight

      result[x][y] = apply(neighbourhood, non_zero, sum_of_weights)

      free(neighbourhood)

cdef void loop_3d(double [:, :, ::1] arr, double [:, ::1] kernel, double [:, ::1] result, int depth, int x_max, int y_max, int kernel_width, double sum_of_weights, int non_zero, f_type apply) nogil:
  cdef int x, y, n, z, offset_x, offset_y
  cdef Neighbourhood * neighbourhood
  
  cdef int x_max_adj = x_max - 1
  cdef int y_max_adj = y_max - 1
  cdef int neighbourhood_size = sizeof(Neighbourhood) * (non_zero * depth)

  cdef Offset * offsets = generate_offsets(kernel, kernel_width, non_zero)

  for x in prange(x_max):
    for y in range(y_max):

      neighbourhood = <Neighbourhood*> malloc(neighbourhood_size) 

      for z in range(depth):
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

          neighbourhood[n].value = arr[z, offset_x, offset_y]
          neighbourhood[n].weight = offsets[n].weight / depth

      result[x][y] = apply(neighbourhood, non_zero, sum_of_weights)

      free(neighbourhood)

cdef f_type func_selector(str func_type):
  if func_type is 'mean': return neighbourhood_sum
  elif func_type is 'variance': return neighbourhood_weighted_variance
  elif func_type is 'standard_deviation': return neighbourhood_weighted_standard_deviation
  elif func_type is 'median': return neighbourhood_weighted_median
  elif func_type is 'q1': return neighbourhood_weighted_q1
  elif func_type is 'q3': return neighbourhood_weighted_q3
  elif func_type is 'mad': return neighbourhood_weighted_mad
  elif func_type is 'mad_std': return neighbourhood_weighted_mad_std
  elif func_type is 'skew_fp': return neighbourhood_weighted_skew_fp
  elif func_type is 'skew_p2': return neighbourhood_weighted_skew_p2
  elif func_type is 'skew_g': return neighbourhood_weighted_skew_g
  elif func_type is 'iqr': return neighbourhood_weighted_iqr
  elif func_type is 'kurtosis': return neighbourhood_weighted_kurtosis_excess
  
  raise Exception('Unable to find filter type!')


def filter_2d(double [:, ::1] arr, double [:, ::1] kernel, str func_type):
  cdef f_type apply = func_selector(func_type)
  cdef int non_zero = np.count_nonzero(kernel)
  cdef double sum_of_weights = np.sum(kernel)
  result = np.empty((arr.shape[0], arr.shape[1]), dtype=np.double)
  cdef double[:, ::1] result_view = result

  loop(arr, kernel, result_view, arr.shape[0], arr.shape[1], kernel.shape[0], sum_of_weights, non_zero, apply)

  return result

def filter_3d(double [:, :, ::1] arr, double [:, ::1] kernel, str func_type):
  cdef f_type apply = func_selector(func_type)
  cdef int non_zero = np.count_nonzero(kernel)
  cdef double sum_of_weights = np.sum(kernel)
  result = np.empty((arr.shape[0], arr.shape[1]), dtype=np.double)
  cdef double[:, ::1] result_view = result

  loop_3d(arr, kernel, result_view, arr.shape[0], arr.shape[1], arr.shape[2], kernel.shape[0], sum_of_weights, non_zero, apply)

  return result