import sys
import os
import numpy as np
from glob import glob
from raster_io import raster_to_array, array_to_raster
from clip_raster import clip_raster


def layers_math(arr_of_rasters, out_raster, method='median'):
    '''
        Calculate descriptive measures across several rasters. Must be the same size.
    '''

    # Memory intense way of doing it. Could refactor to low RAM method.
    if isinstance(arr_of_rasters[0], np.ndarray):
        images_array = np.array(list(map(lambda x: x, arr_of_rasters)))
    else:
        images_array = np.array(list(map(lambda x: raster_to_array(x), arr_of_rasters)))

    if method is 'median':
        return array_to_raster(np.median(images_array, axis=0), out_raster=out_raster, reference_raster=arr_of_rasters[0])
    elif method is 'average':
        return array_to_raster(np.average(images_array, axis=0), out_raster=out_raster, reference_raster=arr_of_rasters[0])
    elif method is 'max':
        return array_to_raster(np.max(images_array, axis=0), out_raster=out_raster, reference_raster=arr_of_rasters[0])
    elif method is 'min':
        return array_to_raster(np.min(images_array, axis=0), out_raster=out_raster, reference_raster=arr_of_rasters[0])

if __name__ == "__main__":
    base = 'C:\\Users\\caspe\\Desktop\\methodology_test\\S1A_IW_GRDH_1SDV_20200210T181801_20200210T181830_031194_03963A_0A4A_NR_Orb_Cal_Spk_TC_dB_Stack.data\\'
    vv = [
        f'{base}Sigma0_VV_db_mst_10Feb2020.img',
        f'{base}Sigma0_VV_db_slv2_04Feb2020.img',
        f'{base}Sigma0_VV_db_slv4_29Jan2020.img',
        f'{base}Sigma0_VV_db_slv6_23Jan2020.img',
        f'{base}Sigma0_VV_db_slv8_17Jan2020.img'
    ]

    # vh = [
    #     f'{base}Sigma0_VH_db_mst_10Feb2020.img',
    #     f'{base}Sigma0_VH_db_slv1_04Feb2020.img',
    #     f'{base}Sigma0_VH_db_slv3_29Jan2020.img',
    #     f'{base}Sigma0_VH_db_slv5_23Jan2020.img',
    #     f'{base}Sigma0_VH_db_slv7_17Jan2020.img'
    # ]

    layers_math(vv, 'C:\\Users\\caspe\\Desktop\\methodology_test\\VV_Min.tif', method='min')
    # layers_math(vh, 'C:\\Users\\caspe\\Desktop\\methodology_test\\VH_Med.tif', method='median')

    # VV_Med = raster_to_array('C:\\Users\\caspe\\Desktop\\methodology_test\\VV_Med.tif')
    # VH_Med = np.multiply(raster_to_array('C:\\Users\\caspe\\Desktop\\methodology_test\\VH_Med.tif'), 0.5637583892617449)

    # Max = array_to_raster(np.max(np.array([VV_Med, VH_Med]), axis=0), out_raster='C:\\Users\\caspe\\Desktop\\methodology_test\\Max.tif', reference_raster='C:\\Users\\caspe\\Desktop\\methodology_test\\VV_Med.tif')