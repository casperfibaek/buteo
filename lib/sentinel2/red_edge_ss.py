import os
import uuid
import shutil
import numpy as np

import sys
sys.path.append('..\\base')
sys.path.append('..\\cython')
sys.path.append('..')

import pyximport
pyximport.install()

from kernel_functions import pansharpen, pansharpen_mad_med
from raster_io import raster_to_array, array_to_raster
from resample import resample

# from orfeo_toolbox import pansharpen


def super_sample_s2(B4, B8, B5=None, B6=None, B7=None, B8A=None,
                    out_folder='../raster/', prefix='', suffix=''):
    paths = {
        'B05': B5,
        'B06': B6,
        'B07': B7,
        'B8A': B8A
    }

    # Sentinel 2 metadata as averages between S2A and S2B
    S2_meta = {
        'B04': {
            'mean': 664.75,      # Central wavelength
            'width': 31.0,       # Width of the spectral band
            'edge_top': 680.25,  # Highest edge on the spectrum
            'edge_bot': 633.75,  # Lowest edge on the spectrum
        },
        'B05': {
            'mean': 703.95,
            'width': 15.5,
            'edge_top': 711.7,
            'edge_bot': 696.2,
        },
        'B06': {
            'mean': 739.8,
            'width': 15.5,
            'edge_top': 747.3,
            'edge_bot': 732.3,
        },
        'B07': {
            'mean': 781.25,
            'width': 20.0,
            'edge_top': 791.25,
            'edge_bot': 771.25,
        },
        'B08': {
            'mean': 832.85,
            'width': 106.0,
            'edge_top': 885.85,
            'edge_bot': 779.85,
        },
    }

    # Radiance p. nm in the bands
    B4_arr = np.divide(raster_to_array(B4), S2_meta['B04']['width'])
    B8_arr = np.divide(raster_to_array(B8), S2_meta['B08']['width'])

    band_array = {
        'B05': np.divide(raster_to_array(B5), S2_meta['B05']['width']) if B5 is not None else False,
        'B06': np.divide(raster_to_array(B6), S2_meta['B05']['width']) if B6 is not None else False,
        'B07': np.divide(raster_to_array(B7), S2_meta['B07']['width']) if B7 is not None else False,
    }

    bands_to_pansharpen = []
    if band_array['B05'] is not False:
        bands_to_pansharpen.append('B05')
    if band_array['B06'] is not False:
        bands_to_pansharpen.append('B06')
    if band_array['B07'] is not False:
        bands_to_pansharpen.append('B07')

    for band_name in bands_to_pansharpen:
        B4_distance = S2_meta[band_name]['edge_bot'] - S2_meta['B04']['edge_top']
        B8_distance = S2_meta['B08']['edge_bot'] - S2_meta[band_name]['edge_top']
        distance_sum = B4_distance + B8_distance
        B4_weight = 1 - (B4_distance / distance_sum)
        B8_weight = 1 - (B8_distance / distance_sum)

        pseudo_band = np.add(
            np.multiply(B4_arr, B4_weight),
            np.multiply(B8_arr, B8_weight),
        )

        resampled_low_res = raster_to_array(resample(paths[band_name], reference_raster=B8)).astype('float64')

        band = band_array[band_name]

        pseudo_band_scaled = np.multiply(
            pseudo_band,
            ((band.mean() / pseudo_band.mean()) * S2_meta[band_name]['width'])
        ).astype('float64')

        pansharpened = pansharpen(resampled_low_res, pseudo_band_scaled, width=3, distance_calc='linear').astype('uint16')
        array_to_raster(pansharpened, reference_raster=B8, out_raster=os.path.join(out_folder, f'{prefix}{band_name}{suffix}.tif'))

    # Special case for Band 8A as no interpolation is necessary. Use band 8.
    if B8A is not None:
        B8_arr = raster_to_array(B8).astype('float64')
        resampled_low_res = raster_to_array(resample(paths[band_name], reference_raster=B8)).astype('float64')
        pansharpened = pansharpen(resampled_low_res, B8_arr, width=3, distance_calc='linear').astype('uint16')
        array_to_raster(pansharpened, reference_raster=B8, out_raster=os.path.join(out_folder, f'{prefix}B8A{suffix}.tif'))


if __name__ == '__main__':
    m10_folder = 'C:\\Users\\caspe\\Desktop\\Data\\Sentinel2\\Egypt\\S2B_MSIL2A_20191019T083949_N0213_R064_T36RTV_20191019T123417.SAFE\\GRANULE\\L2A_T36RTV_A013675_20191019T084241\\IMG_DATA\\R10m\\'
    m20_folder = 'C:\\Users\\caspe\\Desktop\\Data\\Sentinel2\\Egypt\\S2B_MSIL2A_20191019T083949_N0213_R064_T36RTV_20191019T123417.SAFE\\GRANULE\\L2A_T36RTV_A013675_20191019T084241\\IMG_DATA\\R20m\\'

    B4 = m10_folder + 'T36RTV_20191019T083949_B04_10m.jp2'
    B5 = m20_folder + 'T36RTV_20191019T083949_B05_20m.jp2'
    B6 = m20_folder + 'T36RTV_20191019T083949_B06_20m.jp2'
    B7 = m20_folder + 'T36RTV_20191019T083949_B07_20m.jp2'
    B8 = m10_folder + 'T36RTV_20191019T083949_B08_10m.jp2'
    B8A = m20_folder + 'T36RTV_20191019T083949_B8A_20m.jp2'

    super_sample_s2(B4, B8, B5=None, B6=B6, B7=None, B8A=None, out_folder=m10_folder, prefix='', suffix='_pansharpened_3x3_linear')
