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


cdef int compare(const_void *pa, const_void *pb) nogil:
  cdef float a = (<float *> pa)[0]
  cdef float b = (<float *> pb)[0]
  cdef float v = a - b
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


cdef void _calc_stats(float [:] arr, int arr_length, int [:] stats, int stats_length, float [:] result_arr) nogil:
  cdef int x, v, j
  cdef int arr_i, q25i, q50i, q75i
  cdef float q25w, q50w, q75w, q1, q2, q3, iqr, skew_p2, skew_g, skew_r, mad, mad_std
  cdef float * median_deviations

  cdef float v_min = arr[0]
  cdef float v_max = arr[0]
  cdef float v_count = arr_length
  cdef float v_sum = 0
  
  cdef float v_range = 0
  cdef float v_mean = 0
  cdef float variance

  cdef float * ordered_array = <float *> malloc(sizeof(float) * arr_length)
  for v in range(arr_length):
    if arr[v] < v_min:
      v_min = arr[v]
    elif arr[v] > v_max:
      v_max = arr[v]
    v_sum += arr[v]
    ordered_array[v] = arr[v]

  v_range = v_max - v_min
  if v_count == 0:
    v_mean = 0
  else:
    v_mean = v_sum / v_count

  cdef bint calc_medians = 0
  cdef bint calc_mad = 0

  for x in range(stats_length):
    if stats[x] == s_mad or stats[x] == s_mad_std:
      calc_medians = 1
      calc_mad = 1
      break
    if stats[x] == s_med or stats[x] == s_q1 or stats[x] == s_q3 or stats[x] == s_med or stats[x] == s_iqr or stats[x] == s_skew_p2 or stats[x] == s_skew_g or stats[x] == s_skew_r:
      calc_medians = 1

  if calc_medians == 1:
    qsort(<void *> ordered_array, arr_length, sizeof(float), compare)

    arr_i = arr_length - 1
    q25i = <int>(arr_i * 0.25)
    q25w = 1.0 - ((arr_i * 0.25) - q25i)
    q50i = <int>(arr_i * 0.50)
    q50w = 1.0 - ((arr_i * 0.50) - q50i)
    q75i = <int>(arr_i * 0.75)
    q75w = 1.0 - ((arr_i * 0.75) - q75i)

    q1 = ordered_array[q25i] * q25w + ordered_array[q25i + 1] * (1 - q25w)
    q2 = ordered_array[q50i] * q50w + ordered_array[q50i + 1] * (1 - q50w)
    q3 = ordered_array[q75i] * q75w + ordered_array[q75i + 1] * (1 - q75w)
    iqr = q3 - q1
  else:
    arr_i = 0
    q25i = 0
    q25w = 0.0
    q50i = 0
    q50w = 0.0
    q75i = 0
    q75w = 0.0
    q1 = 0.0
    q2 = 0.0
    q3 = 0.0
    iqr = 0.0

  median_deviations = <float *> malloc(sizeof(float) * arr_length)

  cdef float deviations_2 = 0
  cdef float deviations_3 = 0
  cdef float deviations_4 = 0
  for j in range(arr_length):
    if calc_medians == 1:
      median_deviations[j] = fabs(arr[j] - q2)
    deviations_2 += pow(arr[j] - v_mean, 2)
    deviations_3 += pow(arr[j] - v_mean, 3)
    deviations_4 += pow(arr[j] - v_mean, 4)

  if arr_length == 0:
    variance = 0
  else:
    variance = deviations_2 * (1 / <float>arr_length)
  cdef float stdev = sqrt(variance)

  if calc_medians == 1:
    skew_p2 = 3 * ((v_mean - q2) / stdev) if variance != 0 else 0
    skew_g = (q1 + q3 - (2 * q2)) / iqr if iqr != 0 else 0
    skew_r = v_mean / q2 if q2 != 0 else 0
  else:
    skew_p2 = 0
    skew_g = 0
    skew_r = 0

  cdef float skew_fp = (deviations_3 * (1 / <float>arr_length)) / (pow(stdev, 3)) if variance != 0 else 0
  cdef float kurt = (deviations_4 * (1 / <float>arr_length)) / (pow(stdev, 4)) if variance != 0 else 0

  cdef float snr = (v_mean / stdev) if stdev != 0 else 0
  cdef float eff = (variance / pow(v_mean, 2)) if v_mean != 0 else 0
  cdef float cv = (stdev / v_mean) if v_mean != 0 else 0

  if calc_mad == 1:
    qsort(<void *> median_deviations, arr_length, sizeof(float), compare)

    mad = median_deviations[q50i] * q50w + median_deviations[q50i + 1] * (1 - q50w)
    mad_std = mad * 1.4826
  else:
    mad = 0.0
    mad_std = 0.0

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
  med = 14
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


def global_statistics(float [:] arr, translated_stats=[5, 14, 7], stats=None):
  cdef int arr_length = len(arr)
  cdef int stats_length = len(translated_stats) if stats == None else len(stats)

  result = np.empty(stats_length, dtype=np.float)

  cdef int[:] stats_view

  if stats != None:
    stats_translated = np.zeros(stats_length, dtype=np.intc)
    for i, v in enumerate(stats):
      stats_translated[i] = Stats[v].value
    stats_view = stats_translated
  else:
    stats_view = translated_stats

  cdef float[:] result_view = result

  _calc_stats(arr, arr_length, stats_view, stats_length, result_view)

  return result

