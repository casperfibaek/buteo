import numpy as np
import os

import sys
sys.path.append('..\\base')
# sys.path.append('..\\cython')
sys.path.append('..')
sys.path.append('..\\filters')
sys.path.append('..\\..\\OTB-7.0.0-Win64\\lib\\python')

from raster_io import raster_to_array, array_to_raster
from resample import resample
from filters import standardize_filter
import otbApplication


def super_sample_s2(B04_link, B08_link, B05_link=None, B06_link=None, B07_link=None, B8A_link=None,
                    out_folder='../raster/', prefix='', suffix=''):
    assert(isinstance(B05_link, str) or isinstance(B06_link, str) or isinstance(B07_link, str) or isinstance(B8A_link, str))

    paths = {
        'B04': B04_link,
        'B05': B05_link,
        'B06': B06_link,
        'B07': B07_link,
        'B08': B08_link,
        'B8A': B8A_link,
    }

    bands = {
        'B04': raster_to_array(B04_link).astype('float32'),
        'B05': raster_to_array(B05_link).astype('float32') if B05_link is not None else False,
        'B06': raster_to_array(B06_link).astype('float32') if B06_link is not None else False,
        'B07': raster_to_array(B07_link).astype('float32') if B07_link is not None else False,
        'B08': raster_to_array(B08_link).astype('float32'),
        'B8A': raster_to_array(B8A_link).astype('float32') if B8A_link is not None else False,
    }

    bands_to_pansharpen = []
    if bands['B05'] is not False:
        bands_to_pansharpen.append('B05')
    if bands['B06'] is not False:
        bands_to_pansharpen.append('B06')
    if bands['B07'] is not False:
        bands_to_pansharpen.append('B07')
    if bands['B8A'] is not False:
        bands_to_pansharpen.append('B8A')

    for band_x in bands_to_pansharpen:
        if band_x is 'B05':
            pseudo_band = 'B04'
        else:
            pseudo_band = 'B08'
        
        pseudo_path = os.path.join(out_folder, f'{prefix}{band_x}{suffix}_pseudo.tif')
        array_to_raster(bands[pseudo_band], reference_raster=paths[pseudo_band], out_raster=pseudo_path)

        low_res_10m = raster_to_array(resample(paths[band_x], reference_raster=B4)).astype('float32')
        resampled_path = os.path.join(out_folder, f'{prefix}{band_x}{suffix}_resampled.tif')
        array_to_raster(low_res_10m, reference_raster=paths[pseudo_band], out_raster=resampled_path)

        low_res_10m = None

        # pansharpened_path = os.path.join(out_folder, f'{prefix}{band_x}{suffix}_float.tif')
        # pansharpen(pseudo_path, resampled_path, pansharpened_path)

        array_to_raster(raster_to_array(resampled_path).astype('uint16'), reference_raster=paths[pseudo_band], out_raster=os.path.join(out_folder, f'{prefix}{band_x}{suffix}.tif'))
        # array_to_raster(raster_to_array(pansharpened_path).astype('uint16'), reference_raster=paths[pseudo_band], out_raster=os.path.join(out_folder, f'{prefix}{band_x}{suffix}.tif'))

        os.remove(resampled_path)
        os.remove(pseudo_path)
        os.remove(pansharpened_path)


if __name__ == '__main__':
    m10_folder = 'C:\\Users\\caspe\\Desktop\\Data\\Sentinel2\\Egypt\\S2B_MSIL2A_20191019T083949_N0213_R064_T36RTV_20191019T123417.SAFE\\GRANULE\\L2A_T36RTV_A013675_20191019T084241\\IMG_DATA\\R10m\\'
    m20_folder = 'C:\\Users\\caspe\\Desktop\\Data\\Sentinel2\\Egypt\\S2B_MSIL2A_20191019T083949_N0213_R064_T36RTV_20191019T123417.SAFE\\GRANULE\\L2A_T36RTV_A013675_20191019T084241\\IMG_DATA\\R20m\\'

    B4 = m10_folder + 'T36RTV_20191019T083949_B04_10m.jp2'
    B5 = m20_folder + 'T36RTV_20191019T083949_B05_20m.jp2'
    B6 = m20_folder + 'T36RTV_20191019T083949_B06_20m.jp2'
    B7 = m20_folder + 'T36RTV_20191019T083949_B07_20m.jp2'
    B8 = m10_folder + 'T36RTV_20191019T083949_B08_10m.jp2'
    B8A = m20_folder + 'T36RTV_20191019T083949_B8A_20m.jp2'

    # super_sample_s2(B4, B8, B05_link=B5, B06_link=B6, B07_link=None, B8A_link=None, out_folder=m10_folder, prefix='', suffix='_pan')
