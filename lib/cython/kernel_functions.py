import sys
import pyximport
sys.path.append('../base')
pyximport.install()
# from c_filter import filter_2d, filter_median, filter_variance, filter_mad, filter_skew_fp, filter_stdev, filter_skew_p2, filter_skew_g, filter_kurtosis_excess
from filters import mean, variance, standard_deviation, median, median_3d, q3_3d, mad_3d, mad, mad_std, skew_fp, skew_p2, skew_g, kurtosis, iqr
from math import floor, sqrt
from time import time
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from shapely.geometry import Point, Polygon
from raster_io import raster_to_array, array_to_raster


def create_kernel(width, circular=True, weighted_edges=True, holed=False, normalise=True, inverted=False, ring=False, weighted_distance=True, distance_calc='sqrt', distance_scale=True, plot=False, dtype='float32'):
    assert(width % 2 is not 0)

    radius = floor(width / 2) # 4
    kernel = np.zeros((width, width), dtype=dtype)
    dist_offset = 0.2

    for x in range(width):
        for y in range(width):
            xm = x - radius
            ym = y - radius

            if holed is True and xm is 0 and ym is 0:
                weight = 0
                continue

            dist = sqrt(pow(xm, 2) + pow(ym, 2))

            weight = 1

            if weighted_distance is True:
                if xm is 0 and ym is 0:
                    weight = 1
                else:
                    if distance_scale is True:
                        scale = sqrt((radius ** 2) + (radius ** 2)) + sqrt(0.5)
                        if ring is True:
                            scale_2 = scale / 2
                            abs_dist = abs(dist - scale_2)
                            if distance_calc is 'sqrt':
                                weight = 1 - (sqrt(abs_dist) / sqrt(scale_2))
                            if distance_calc is 'pow':
                                weight = 1 - (pow(abs_dist, 2) / pow(scale_2, 2))
                            if distance_calc is 'linear':
                                weight = 1 - (abs_dist / scale_2)
                        else:
                            if distance_calc is 'sqrt':
                                weight = 1 - (sqrt(dist) / sqrt(scale))
                            if distance_calc is 'pow':
                                weight = 1 - (pow(dist, 2) / pow(scale, 2))
                            if distance_calc is 'linear':
                                weight = 1 - (dist / scale)
                    else:
                        if distance_calc is 'sqrt':
                            weight = 1 / (sqrt(dist) + dist_offset)
                        if distance_calc is 'pow':
                            weight = 1 / (pow(dist, 2) + dist_offset)
                        if distance_calc is 'linear':
                            weight = 1 / dist

            if circular is True:
                if weighted_edges is False:
                    if (dist - radius) > sqrt(0.5):
                        weight = 0
                else:
                    circle = Point(0, 0).buffer(radius + 0.5)
                    polygon = Polygon([(xm - 0.5, ym - 0.5), (xm - 0.5, ym + 0.5), (xm + 0.5, ym + 0.5), (xm + 0.5, ym - 0.5)])
                    intersection = polygon.intersection(circle)

                    # Area of a pixel is 1, no need to normalise.
                    weight = weight * intersection.area

            kernel[x][y] = weight if inverted is False else 1 - weight

    if normalise is True:
        kernel = np.divide(kernel, kernel.sum())

    if plot is True:
        fig, ax = plt.subplots()

        if normalise is False:
            im = ax.imshow(kernel, cmap='bone_r', interpolation=None, vmin=0, vmax=1)
        else:
            im = ax.imshow(kernel, cmap='bone_r', interpolation=None)

        circle = plt.Circle((radius, radius), radius + 0.5, color='black', fill=False, linestyle='--')
        ax.add_artist(circle)
        ax.invert_yaxis()

        for i in range(width):
            for j in range(width):
                ax.text(j, i, f"{kernel[i, j]:.4f}", ha="center", va="center", color="tomato")

        plt.figtext(0.5, 0.025, f"size: {width}x{width}, weighted_edges: {weighted_edges}, weighted_distance: {weighted_distance}, method: {distance_calc}", ha="center")

        plt.colorbar(im)
        plt.show()

    return kernel


# def pansharpen(low, high, width=3, distance_calc='linear'):
#     kernel = create_kernel(width, circular=True, weighted_edges=True, normalise=True, weighted_distance=True, distance_scale=True, distance_calc=distance_calc, plot=False).astype(np.float)
#     low_local_mean = filter_2d(low, kernel)
#     low_stdev = np.sqrt(filter_variance(low, kernel))
#     high_local_mean = filter_2d(high, kernel)
#     high_stdev = np.sqrt(filter_variance(high, kernel))

#     return (((high - high_local_mean) * low_stdev) / (high_stdev)) + low_local_mean


# def pansharpen_mad_med(low, high, width=3, distance_calc='linear'):
#     kernel = create_kernel(width, circular=True, weighted_edges=True, normalise=True, weighted_distance=True, distance_scale=True, distance_calc=distance_calc, plot=False).astype(np.float)
#     low_local_med = filter_median(low, kernel)
#     low_stdev = filter_mad(low, kernel)
#     high_local_med = filter_median(high, kernel)
#     high_stdev = filter_mad(high, kernel)

#     return (((high - high_local_med) * low_stdev) / (high_stdev)) + low_local_med


if __name__ == "__main__":
    kernel_size = 21
    # kernel = create_kernel(kernel_size, circular=True, weighted_edges=True, inverted=False, ring=False, holed=False, normalise=True, weighted_distance=True, distance_scale=True, distance_calc='linear', plot=False).astype(np.float)
    kernel_sum = create_kernel(kernel_size, circular=True, weighted_edges=True, inverted=False, ring=False, holed=False, normalise=False, weighted_distance=False, distance_scale=True, distance_calc='linear', plot=False).astype(np.float)
    # kernel_holed = create_kernel(kernel_size, circular=True, weighted_edges=True, inverted=False, ring=False, holed=True, normalise=True, weighted_distance=True, distance_scale=True, distance_calc='linear', plot=False).astype(np.float)
    # kernel_ring = create_kernel(kernel_size, circular=True, weighted_edges=True, inverted=False, ring=True, holed=True, normalise=True, weighted_distance=True, distance_scale=True, distance_calc='linear', plot=False).astype(np.float)

    # folder = 'C:\\Users\\caspe\\Desktop\\Data\\Sentinel2\\'
    folder_s1_grd = 'C:\\Users\\caspe\\Desktop\\Data\\Sentinel1\\S1B_IW_GRDH_1SDV_20200204T181721_20200204T181746_020123_02616A_B5C4_Orb_NR_Cal_TF_TC_Stack.data\\'
    folder_s1 = 'C:\\Users\\caspe\\Desktop\\Data\\Sentinel1\\'
    # in_raster_coh = f'{folder_s1}coherence_accra.tif'
    # in_raster_s1 = f'{folder}accra_s1.tif'
    # in_raster_b4 = f'{folder}accra_b4.jp2'
    # in_raster_b8 = f'{folder}accra_b8.jp2'
    # out_raster_mad_b4 = f'{folder}accra_b4_{kernel_size}x{kernel_size}_mad-std.tif'
    # out_raster_mad_b8 = f'{folder}accra_b8_{kernel_size}x{kernel_size}_mad-std.tif'
    # out_raster_meddev_b4 = f'{folder}accra_b4_{kernel_size}x{kernel_size}_med-dev.tif'
    # out_raster_meddev_b8 = f'{folder}accra_b8_{kernel_size}x{kernel_size}_med-dev.tif'
    # out_raster_ndvi = f'{folder}accra_ndvi_i.tif'
    # out_raster_s1 = f'{folder}accra_med_5x5.tif'
    # out_raster_coh = f'{folder_s1}coherence_accra_pow_med_5x5.tif'

    # b4 = raster_to_array(in_raster_b4).astype(np.float)
    # b8 = raster_to_array(in_raster_b8).astype(np.float)
    # s1 = raster_to_array(in_raster_s1).astype(np.float)
    # coh = raster_to_array(in_raster_coh).astype(np.float)

    surf = raster_to_array(f'{folder_s1}surf_v2.tif').astype(np.float)
    # coh = raster_to_array(f'{folder_s1}coherence_accra.tif').astype(np.float)
    # grd_1 = raster_to_array(f'{folder_s1_grd}Gamma0_VV_mst_04Feb2020.img').astype(np.float)
    # grd_2 = raster_to_array(f'{folder_s1_grd}Gamma0_VV_slv1_10Feb2020.img').astype(np.float)
    # stack = np.array([grd_1,  grd_2])

    before2 = time()
    # array_to_raster(median_3d(stack, kernel).astype('float32'), out_raster=f'{folder_s1}backscatter_3x3-median.tif', reference_raster=f'{folder_s1_grd}Gamma0_VV_mst_04Feb2020.img', dst_nodata=None)
    # array_to_raster(median(coh, kernel).astype('float32'), out_raster=f'{folder_s1}coherence_7x7-median.tif', reference_raster=f'{folder_s1}coherence_accra.tif', dst_nodata=None)
    # array_to_raster(mad_3d(stack, kernel).astype('float32'), out_raster=f'{folder_s1}grd_5x5-3d-mad.tif', reference_raster=f'{folder_s1}Gamma0_VV_mst_04Feb2020.img', dst_nodata=None)
    # array_to_raster((((b4 - b8) / (b4 + b8)) + 1).astype('float32'), out_raster=out_raster_ndvi, reference_raster=in_raster_b4, dst_nodata=None)
    # array_to_raster(mad(b8, kernel).astype('float32'), out_raster=out_raster_mad_b8, reference_raster=in_raster_b8, dst_nodata=None)
    array_to_raster(mean(surf, kernel_sum).astype('float32'), out_raster=f'{folder_s1}surf_v2_100m-density.tif', reference_raster=f'{folder_s1}surf_v2.tif', dst_nodata=None)
    # array_to_raster(mean(np.abs(median(b8, kernel_holed) - b8), kernel).astype('float32'), out_raster=out_raster_meddev_b8, reference_raster=in_raster_b4, dst_nodata=None)
    # array_to_raster(np.power(median(coh, kernel), 2).astype('float32'), out_raster=out_raster_coh, reference_raster=in_raster_coh, dst_nodata=None)
    print(time() - before2)
