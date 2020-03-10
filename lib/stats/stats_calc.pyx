# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, profile = False
cimport cython
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, pow, fabs
import numpy as np
from enum import Enum

cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size, int(*compar)(const_void *, const_void *)) nogil

cdef struct IndexedElement:
    int index
    double value

cdef int compare(const_void *pa, const_void *pb) nogil:
  cdef double a = (<double *> pa)[0]
  cdef double b = (<double *> pb)[0]
  cdef double v = a - b
  if v < 0: return -1
  elif v > 0: return 1
  return 0

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
  s_snr = 19
  s_cv = 20
  s_eff = 21


# TODO: q1, q3 not working

cdef void _calc_stats(double [:] arr, int arr_length, int [:] stats, int stats_length, double [:] result_arr):
  cdef int x, v, j

  cdef double v_min = arr[0]
  cdef double v_max = arr[0]
  cdef double v_count = arr_length
  cdef double v_sum = 0
  
  cdef double v_range = 0
  cdef double v_mean = 0

  cdef double * ordered_array = <double *> malloc(sizeof(double) * arr_length)
  for v in range(arr_length):
    if arr[v] < v_min:
      v_min = arr[v]
    elif arr[v] > v_max:
      v_max = arr[v]
    v_sum += arr[v]
    ordered_array[v] = arr[v]

  v_range = v_max - v_min
  v_mean = v_sum / v_count

  qsort(<void *> ordered_array, arr_length, sizeof(double), compare)

  cdef int arr_i = arr_length - 1
  cdef int q25i = <int>(arr_i * 0.25)
  cdef double q25w = 1.0 - ((arr_i * 0.25) - q25i)
  cdef int q50i = <int>(arr_i * 0.50)
  cdef double q50w = 1.0 - ((arr_i * 0.50) - q50i)
  cdef int q75i = <int>(arr_i * 0.75)
  cdef double q75w = 1.0 - ((arr_i * 0.75) - q75i)

  cdef double q1 = ordered_array[q25i] * q25w + ordered_array[q25i + 1] * (1 - q25w)
  cdef double q2 = ordered_array[q50i] * q50w + ordered_array[q50i + 1] * (1 - q50w)
  cdef double q3 = ordered_array[q75i] * q75w + ordered_array[q75i + 1] * (1 - q75w)

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

  cdef double variance = deviations_2 * (1 / <double>arr_length)
  cdef double stdev = sqrt(variance)

  cdef double skew_fp = (deviations_3 * (1 / <double>arr_length)) / (pow(stdev, 3))
  cdef double skew_p2 = 3 * ((v_mean - q2) / stdev)
  cdef double skew_g = 0 if iqr == 0 else (q1 + q3 - (2 * q2)) / iqr
  cdef double skew_r = v_mean / q2

  cdef double kurt = (deviations_4 * (1 / <double>arr_length)) / (pow(stdev, 4))

  qsort(<void *> median_deviations, arr_length, sizeof(double), compare)

  cdef double mad = median_deviations[q50i] * q50w + median_deviations[q50i + 1] * (1 - q50w)
  cdef double mad_std = mad * 1.4826

  cdef double snr = v_mean / stdev
  cdef double eff = variance / pow(v_mean, 2)
  cdef double cv = stdev / v_mean

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
      result_arr[x] = variance
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
    elif(stats[x] == s_snr):
      result_arr[x] = snr
    elif(stats[x] == s_eff):
      result_arr[x] = eff
    elif(stats[x] == s_cv):
      result_arr[x] = cv

class Stats(Enum):
  min = 1
  max = 2
  count = 3
  range = 4
  mean = 5
  var = 6
  variance = 6
  std = 7
  stdev = 7
  standard_deviation = 7
  skew_fp = 8
  skew_p2 = 9
  skew_g = 10
  skew_r = 11
  kurt = 12
  kurtosis = 12
  q1 = 13
  median = 14
  q2 = 14
  q3 = 15
  iqr = 16
  mad = 17
  mad_std = 18
  mad_stdev = 18
  mad_standard_deviation = 18
  signal_to_noise = 19
  snr = 19
  cv = 20
  coefficient_of_variation = 20
  eff = 21
  efficiency = 21


def enumerate_stats(stats_names=['mean', 'median', 'stdev']):
  arr = np.zeros(len(stats_names), dtype=np.intc)
  for i, v in enumerate(stats_names):
    arr[i] = Stats[v].value
  return arr


def global_statistics(double [:] arr, translated_stats=[5, 14, 7], stats=None):
  cdef int arr_length = len(arr)
  cdef int stats_length = len(translated_stats) if stats == None else len(stats)

  result = np.empty(stats_length, dtype=np.double)

  cdef int[:] stats_view

  if stats != None:
    stats_translated = np.zeros(stats_length, dtype=np.intc)
    for i, v in enumerate(stats):
      stats_translated[i] = Stats[v].value
    stats_view = stats_translated
  else:
    stats_view = translated_stats

  cdef double[:] result_view = result

  _calc_stats(arr, arr_length, stats_view, stats_length, result_view)

  return result
