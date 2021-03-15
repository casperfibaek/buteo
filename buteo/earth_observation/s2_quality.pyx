# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, profile=False
cimport cython
from cython.parallel cimport prange


cdef struct Offset:
  int x
  int y
  double weight


cdef double assess_quality_func(
    int scl,
    int b2,
    int b12,
    double band_cldprb,
    double darkprb,
    int aot,
    int nodata,
    double time_difference,
    double sun_elevation,
) nogil:
    cdef double cld = 0
    cdef double quality = 0.0

    if nodata == 1: # SC_NODATA
        quality = -10.0
    elif scl == 1:  # SC_SATURATED_DEFECTIVE
        quality = -10.0
    elif scl == 2:  # SC_DARK_FEATURE_SHADOW
        quality = 55.0
    elif scl == 3:  # SC_CLOUD_SHADOW
        quality = 45.0
    elif scl == 4:  # SC_VEGETATION
        quality = 95.0
    elif scl == 5:  # SC_NOT_VEGETATED
        quality = 95.0
    elif scl == 6:  # SC_WATER
        quality = 90.0
    elif scl == 7:  # SC_UNCLASSIFIED
        quality = 80.0
    elif scl == 8:  # SC_CLOUD_MEDIUM_PROBA
        quality = 25.0
    elif scl == 9:  # SC_CLOUD_HIGH_PROBA
        quality = 10.0
    elif scl == 10: # SC_THIN_CIRRUS
        quality = 70.0
    elif scl == 11: # SC_SNOW_ICE
        quality = 55.0

    # First evaluate Aerosol Optical Thickness
    quality = quality + ((-0.01 * aot) + 2.5)

    # Evalutate cloud percentage
    if ((scl == 4) | (scl == 5) | (scl == 6) | (scl == 7) | (scl == 11)):
        if (b12 < 1000):
            cld = band_cldprb + ((0.01 * b12) - 10)
        else:
            cld = band_cldprb

    quality = quality - (cld * 2)

    # Evaluate dark areas
    quality = quality - (darkprb * 2)

    # Evaluate blue band
    if b2 > 700:
        quality = quality + (-0.0175 * b2 + 7)
    elif b2 < 100:
        quality = quality + ((-0.002 * (b2 * b2)) + (0.4 * b2) - 20)
    
    # Evauluate time difference: minus 0.5% quality per week
    quality = quality + (-0.0725 * time_difference)

    # Evaluate sun elevation, higher is better
    # +5% quality for sun at zenith, -10% for sun at horizon
    quality = quality + ((-0.0012 * (sun_elevation * sun_elevation)) + (0.2778 * sun_elevation) - 10)

    if quality <= 0:
        if nodata == 1 or scl == 1:
            quality = -10.0
        else:
            quality = 0

    return quality


cdef double assess_quality(
    int [:, ::1] scl,
    int [:, ::1] band_02,
    int [:, ::1] band_12,
    double [:, ::1] band_cldprb,
    double [:, ::1] darkprb,
    int [:, ::1] aot,
    int [:, ::1] nodata_dilated,
    double [:, ::1] quality,
    double time_difference,
    double sun_elevation,
    int x_max,
    int y_max,
) nogil:
    cdef int x, y
    cdef double quality_sum = 0

    for x in prange(x_max, nogil=True):
        for y in prange(y_max):
            quality[x][y] = assess_quality_func(scl[x][y], band_02[x][y], band_12[x][y], band_cldprb[x][y], darkprb[x][y], aot[x][y], nodata_dilated[x][y], time_difference, sun_elevation)
            quality_sum += quality[x][y]
    
    return quality_sum

cpdef double radiometric_quality(scl, band_02, band_12, band_cldprb, darkprb, aot, nodata_dilated, quality, td, sun_elevation):
    cdef int [:, ::1] scl_view = scl
    cdef int [:, ::1] band_02_view = band_02
    cdef int [:, ::1] band_12_view = band_12
    cdef double [:, ::1] band_cldprb_view = band_cldprb
    cdef double [:, ::1] darkprb_view = darkprb
    cdef int [:, ::1] aot_view = aot
    cdef int [:, ::1] nodata_dilated_view = nodata_dilated
    cdef double [:, ::1] quality_view = quality

    cdef int x_max = scl.shape[0]
    cdef int y_max = scl.shape[1]
    
    return assess_quality(
        scl_view,
        band_02_view,
        band_12_view,
        band_cldprb_view,
        darkprb_view,
        aot_view,
        nodata_dilated_view,
        quality_view,
        td,
        sun_elevation,
        x_max,
        y_max,
    )


def feather_s2_filter(tracking_image, value_to_count, width=21):
    kernel = create_kernel(
        width,
        circular=circular,
        holed=False,
        normalise=False,
        weighted_edges=False,
        weighted_distance=False,
    )
    return feather_s2_array(tracking_image, value_to_count, kernel)


cdef double count_occurrances(
    int * neighbourhood,
    int [::1] values_to_count,
    int val_to_count_len,
    int non_zero,
) nogil:
    cdef int n, m
    cdef int count = 0

    for n in range(non_zero):
        for m in range(val_to_count_len):
            if neighbourhood[n] == values_to_count[m]:
                count += 1
    
    return <double> count / <double> non_zero


cdef void loop_feather(
    int [:, ::1] tracker,
    double [:, ::1] kernel,
    double [:, ::1] result,
    int x_max,
    int y_max,
    int kernel_width,
    int non_zero,
    int [::1] values_to_count,
    int val_to_count_len,
) nogil:
    cdef int x, y, n, m, offset_x, offset_y
    cdef int x_max_adj = x_max - 1
    cdef int y_max_adj = y_max - 1
    cdef int * neighbourhood,
    cdef int neighbourhood_size = sizeof(int) * non_zero

    cdef Offset * offsets = generate_offsets(kernel, kernel_width, non_zero)

    for x in prange(x_max):
        for y in prange(y_max):

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

                neighbourhood[n] = tracker[offset_x][offset_y]
            
            result[x][y] = count_occurrances(neighbourhood, values_to_count, val_to_count_len, non_zero)

            free(neighbourhood)


def feather_s2_array(tracker, values_to_count, kernel):
    cdef int non_zero = np.count_nonzero(kernel)
    cdef int [::1] val_to_count_view = values_to_count
    cdef int val_to_count_len = len(values_to_count)
    cdef double [:, ::1] kernel_view = kernel.astype(np.double) if kernel.dtype != np.double else kernel

    result = np.zeros(tracker.shape, dtype=np.double)
    cdef double [:, ::1] result_view = result
    cdef int [:, ::1] tracker_view = tracker.astype(np.intc) if tracker.dtype != np.intc else tracker

    loop_feather(tracker_view, kernel_view, result_view, tracker.shape[0], tracker.shape[1], kernel.shape[0], non_zero, val_to_count_view, val_to_count_len)

    return result
