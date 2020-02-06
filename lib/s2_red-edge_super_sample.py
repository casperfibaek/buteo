import os
import uuid
import shutil
import numpy as np

from raster_io import raster_to_array, array_to_raster
from resample import resample
from orfeo_toolbox import pansharpen
from mask_raster import mask_raster as mask_raster_func


def super_sample_s2(B4, B8, B5=None, B6=None, B7=None, B8A=None,
                    out_folder='../raster/', mask_raster=None, prefix='', suffix=''):
    # Create temp folder
    temp_folder_name = os.path.join('../temp/', uuid.uuid4().hex)
    temp_folder_path = os.path.abspath(temp_folder_name)
    os.makedirs(temp_folder_name)

    # Wrap all in a try catch to ensure deletion of the temp folder
    try:
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

        band_arrays = {
            'B05': np.divide(raster_to_array(B5), S2_meta['B05']['width']) if B5 is not None else False,
            'B06': np.divide(raster_to_array(B6), S2_meta['B05']['width']) if B6 is not None else False,
            'B07': np.divide(raster_to_array(B7), S2_meta['B07']['width']) if B7 is not None else False,
        }

        bands_to_pansharpen = []
        if band_arrays['B05'] is not False:
            bands_to_pansharpen.append('B05')
        if band_arrays['B06'] is not False:
            bands_to_pansharpen.append('B06')
        if band_arrays['B07'] is not False:
            bands_to_pansharpen.append('B07')

        for band in bands_to_pansharpen:
            B4_distance = S2_meta[band]['edge_bot'] - S2_meta['B04']['edge_top']
            B8_distance = S2_meta['B08']['edge_bot'] - S2_meta[band]['edge_top']
            distance_sum = B4_distance + B8_distance
            B4_weight = 1 - (B4_distance / distance_sum)
            B8_weight = 1 - (B8_distance / distance_sum)

            ratio = np.add(
                np.multiply(B4_arr, B4_weight),
                np.multiply(B8_arr, B8_weight),
            )

            resampled_name = os.path.join(temp_folder_path, f'{band}_resampled_10m.tif')
            resample(paths[band], reference_raster=B8, out_raster=resampled_name)

            if mask_raster is not None:
                resampled_name_masked = os.path.join(temp_folder_path, f'{band}_resampled_masked_10m.tif')
                mask_raster_func(resampled_name, mask_raster=mask_raster, out_raster=resampled_name_masked)
                resampled_name = resampled_name_masked

            band_array = band_arrays[band]

            pan = np.multiply(
                ratio,
                (band_array.mean() / ratio.mean() * S2_meta[band]['width'])
            ).astype('uint16')
            pan_name = os.path.join(temp_folder_path, f'{band}_pan_ratio_10m.tif')
            array_to_raster(pan, reference_raster=B8, out_raster=pan_name)

            out_name = os.path.join(out_folder, f'{prefix}{band}{suffix}.tif')

            pansharpen(pan_name, resampled_name, out_name, out_datatype='uint16')

        # Special case for Band 8A as no interpolation is necessasry.
        if B8A is not None:
            B8_arr = raster_to_array(B8A)
            resampled_name = os.path.join(temp_folder_path, 'B8A_resampled_10m.tif')
            resample(paths[band], reference_raster=B8, out_raster=resampled_name)

            if mask_raster is not None:
                resampled_name_masked = os.path.join(temp_folder_path, 'B8A_resampled_masked_10m.tif')
                mask_raster_func(resampled_name, mask_raster=mask_raster, out_raster=resampled_name_masked)
                resampled_name = resampled_name_masked

            out_name_B8A = os.path.join(out_folder, f'{prefix}B8A{suffix}.tif')
            pansharpen(B8, resampled_name, out_name_B8A, out_datatype='uint16')
    finally:
        shutil.rmtree(temp_folder_path)


if __name__ == '__main__':
    m10_folder = 'E:\\sentinel_2_data\\bornholm_mosaic\\S2A_MSIL2A_20190514T101031_N0212_R022_T33UWB_20190514T134657.SAFE\\GRANULE\\L2A_T33UWB_A020325_20190514T101441\\IMG_DATA\\R10m\\'
    m20_folder = 'E:\\sentinel_2_data\\bornholm_mosaic\\S2A_MSIL2A_20190514T101031_N0212_R022_T33UWB_20190514T134657.SAFE\\GRANULE\\L2A_T33UWB_A020325_20190514T101441\\IMG_DATA\\R20m\\'

    B4 = m10_folder + 'T33UWB_20190514T101031_B04_10m.jp2'
    B5 = m20_folder + 'T33UWB_20190514T101031_B05_20m.jp2'
    B6 = m20_folder + 'T33UWB_20190514T101031_B06_20m.jp2'
    B7 = m20_folder + 'T33UWB_20190514T101031_B07_20m.jp2'
    B8 = m10_folder + 'T33UWB_20190514T101031_B08_10m.jp2'
    B8A = m20_folder + 'T33UWB_20190514T101031_B8A_20m.jp2'

    super_sample_s2(B4, B8, B5=B5, B6=B6, B7=B7, B8A=B8A, out_folder=m10_folder, prefix='ss_')