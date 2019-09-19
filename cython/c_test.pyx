# cython: warn.undeclared=True, warn.unused=True, profile=True, linetrace=True
# distutils: extra_compile_args=-openmp
# distutils: extra_link_args=-openmp

cimport cython
from cython.parallel import prange
import numpy as np


def compute1(int [:, ::1] arr):
    cdef Py_ssize_t x, y, n, m, x_max, y_max
    n = 5
    x_max = 10
    y_max = 10

    cdef int aggregated
    for x in prange(x_max, nogil=True):
        for y in range(y_max):
            aggregated = 0

            for m in range(n):
                aggregated += m

            arr[x, y] = aggregated

    return aggregated

# def compute2(int[:, ::1] array_1):
#     cdef Py_ssize_t x_max = array_1.shape[0]
#     cdef Py_ssize_t y_max = array_1.shape[1]

#     result = np.zeros((x_max, y_max), dtype=np.int)
#     cdef int[:, ::1] result_view = result


#     cdef Py_ssize_t x, y

#     cdef int aggregated = 0
#     for x in prange(x_max, nogil=True):
#         for y in range(y_max):
#             aggregated += 1
#             result_view[x, y] = array_1[x, y]

#     return result