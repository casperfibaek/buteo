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

cdef struct IndexedElement:
    int index
    double value

cdef int _compare(const_void *a, const_void *b) nogil:
  cdef double v = (<IndexedElement*> a).value - (<IndexedElement*> b).value
  if v < 0: return -1
  elif v > 0: return 1
  else: return 0

# cdef void argsort(Neighbourhood * neighbourhood, int* order, int non_zero) nogil:
#   cdef int i
  
#   # Allocate index tracking array.
#   cdef IndexedElement* order_struct = <IndexedElement *> malloc(non_zero * sizeof(IndexedElement))
  
#   # Copy data into index tracking array.
#   for i in range(non_zero):
#       order_struct[i].index = i
#       order_struct[i].value = neighbourhood[i].value
      
#   # Sort index tracking array.
#   qsort(<void *> order_struct, non_zero, sizeof(IndexedElement), _compare)
  
#   # Copy indices from index tracking array to output array.
#   for i in range(non_zero):
#       order[i] = order_struct[i].index
      
#   # Free index tracking array.
#   free(order_struct)

ctypedef enum stat_name:
  s_min = 1
  s_max = 2
  s_count = 3
  s_range = 4
  s_mean = 5
  s_var = 6
  s_std = 7
  s_skew_fp = 8
  s_skew_p2 = 9
  s_skew_g = 10
  s_skew_r = 11
  s_kurt = 12
  s_q1 = 13
  s_med = 14
  s_q3 = 15
  s_iqr = 16
  s_mad = 17
  s_mad_std = 18


# TODO: Sort function is broken
# TODO: Stdev not working
# TODO: Seperate translate function

cdef void _calc_stats(double [:] arr, int arr_length, int [:] stats, int stats_length, double [:] result_arr):
  cdef int x, v, j

  cdef double v_min = arr[0]
  cdef double v_max = arr[0]
  cdef double v_count = arr_length
  cdef double v_sum = 0
  
  cdef double v_range = 0
  cdef double v_mean = 0

  cdef double* ordered_array = <IndexedElement *> malloc(sizeof(IndexedElement) * arr_length)
  for v in range(arr_length):
    if arr[v] < v_min:
      v_min = arr[v]
    elif arr[v] > v_max:
      v_max = arr[v]
    v_sum += arr[v]
    ordered_array[v].index = index
    ordered_array[v].value = arr[v]

  v_range = v_max - v_min
  v_mean = v_sum / v_count

  qsort(<void *> ordered_array, arr_length, sizeof(double), _compare)

  for q in range(arr_length):
    print(ordered_array[q])

  # calculate median and quintiles
  cdef int bq25 = <int>(arr_length * 0.25)
  cdef double bq25w = bq25 % (arr_length * 0.25)
  cdef int bq50 = <int>(arr_length * 0.50)
  cdef double bq50w = bq50 % (arr_length * 0.50)
  cdef int bq75 = <int>(arr_length * 0.75)
  cdef double bq75w = bq75 % (arr_length * 0.75)

  cdef double q1 = ordered_array[bq25] * bq25w + ordered_array[bq25 + 1] * (1 - bq25w)
  cdef double q2 = ordered_array[bq50] * bq50w + ordered_array[bq50 + 1] * (1 - bq50w)
  cdef double q3 = ordered_array[bq75] * bq75w + ordered_array[bq75 + 1] * (1 - bq75w)

  cdef double iqr = q3 - q1

  cdef double * median_deviations = <double *> malloc(sizeof(double) * arr_length)

  cdef double deviations_2 = 0
  cdef double deviations_3 = 0
  cdef double deviations_4 = 0
  for j in range(arr_length):
    median_deviations[j] = fabs(arr[j] - q2)
    deviations_2 += pow(arr[j] - v_mean, 2)
    deviations_3 += pow(arr[j] - v_mean, 3)
    deviations_4 += pow(arr[j] - v_mean, 4)

  cdef double var = deviations_2 * (1 / arr_length)
  cdef double stdev = sqrt(var)

  cdef double skew_fp = (deviations_3 * (1 / arr_length)) / (pow(stdev, 3))
  cdef double skew_p2 = 3 * ((v_mean - q2) / stdev)
  cdef double skew_g = 0 if iqr == 0 else (q1 + q3 - (2 * q2)) / iqr
  cdef double skew_r = v_mean / q2

  cdef double kurt = (deviations_4 * (1 / arr_length)) / (pow(stdev, 4))

  qsort(<void *> median_deviations, arr_length, sizeof(double), _compare)

  cdef double mad = median_deviations[bq50] * bq50w + median_deviations[bq50 + 1] * (1 - bq50w)
  cdef double mad_std = mad * 1.4826

  free(ordered_array)
  free(median_deviations)

  for x in range(stats_length):
    if(stats[x] == s_min):
      result_arr[x] = v_min
    elif(stats[x] == s_max):
      result_arr[x] = v_max
    elif(stats[x] == s_count):
      result_arr[x] = v_count
    elif(stats[x] == s_range):
      result_arr[x] = v_range
    elif(stats[x] == s_mean):
      result_arr[x] = v_mean
    elif(stats[x] == s_var):
      result_arr[x] = var
    elif(stats[x] == s_std):
      result_arr[x] = stdev
    elif(stats[x] == s_skew_fp):
      result_arr[x] = skew_fp
    elif(stats[x] == s_skew_p2):
      result_arr[x] = skew_p2
    elif(stats[x] == s_skew_g):
      result_arr[x] = skew_g
    elif(stats[x] == s_skew_r):
      result_arr[x] = skew_r
    elif(stats[x] == s_kurt):
      result_arr[x] = kurt
    elif(stats[x] == s_q1):
      result_arr[x] = q1
    elif(stats[x] == s_med):
      result_arr[x] = q2
    elif(stats[x] == s_q3):
      result_arr[x] = q3
    elif(stats[x] == s_iqr):
      result_arr[x] = iqr
    elif(stats[x] == s_mad):
      result_arr[x] = mad
    elif(stats[x] == s_mad_std):
      result_arr[x] = mad_std


def global_statistics(double [:] arr, stats=['mean', 'median', 'stdev']):
  cdef int arr_length = len(arr)
  cdef int stats_length = len(stats)

  result = np.empty(len(stats), dtype=np.double)
  stats_translated = np.zeros(len(stats), dtype=np.intc)

  for i, v in enumerate(stats):
    if v == 'min':
      stats_translated[i] = 1
    elif v == 'max':
      stats_translated[i] = 2
    elif v == 'count':
      stats_translated[i] = 3
    elif v == 'range':
      stats_translated[i] = 4
    elif v == 'mean':
      stats_translated[i] = 5
    elif v == 'var':
      stats_translated[i] = 6
    elif v == 'stdev':
      stats_translated[i] = 7
    elif v == 'skew_fp':
      stats_translated[i] = 8
    elif v == 'skew_p2':
      stats_translated[i] = 9
    elif v == 'skew_g':
      stats_translated[i] = 10
    elif v == 'skew_r':
      stats_translated[i] = 11
    elif v == 'kurtosis':
      stats_translated[i] = 12
    elif v == 'q1':
      stats_translated[i] = 13
    elif v == 'median':
      stats_translated[i] = 14
    elif v == 'q3':
      stats_translated[i] = 15
    elif v == 'iqr':
      stats_translated[i] = 16
    elif v == 'mad':
      stats_translated[i] = 17
    elif v == 'mad_std':
      stats_translated[i] = 18

  cdef double[:] result_view = result
  cdef int[:] stats_view = stats_translated

  _calc_stats(arr, arr_length, stats_view, stats_length, result_view)

  return result
