import unittest
import gdal
import sys
import psutil
import numpy as np

sys.path.append('../lib')
from array_to_raster import array_to_raster
from raster_to_array import raster_to_array
from clip_raster import clip_raster
from orfeo_toolbox import local_stats, haralick, concatenate_images, dimension_reduction, rescale

raster_10m = './T32VNJ_20180727T104021_B04_10m.jp2'
b5 = './T32VNJ_20180727T104021_B05_ss_10m.tif'
b6 = './T32VNJ_20180727T104021_B06_ss_10m.tif'
b7 = './T32VNJ_20180727T104021_B07_ss_10m.tif'
raster_100m = './raster_100m.tif'
raster_200m = './raster_200m.tif'
geom_roses = './roses.geojson'
geom_partial = './withinPartial.geojson'
geom_aalborg = './aalborg.geojson'
arr_100 = np.random.randint(0, high=100, size=(100, 100))
arr_200 = np.random.randint(0, high=100, size=(200, 200))


class Test_suite(unittest.TestCase):
    ''' ****************************************
        array_to_raster
    **************************************** '''
    def test_array_to_raster_no_reference(self):
        with self.assertRaises(AttributeError):
            array_to_raster(arr_100)

    def test_array_to_raster_manual_reference(self):
        raster = array_to_raster(
            arr_100,
            top_left=[588550, 6141260],
            pixel_size=[100, 100],
            projection="+proj=utm +zone=32 +ellps=WGS84 +datum=WGS84 +units=m +no_defs",
        )
        self.assertEqual(isinstance(raster, gdal.Dataset), True, "Should be a GDAL dataframe")

    def test_array_to_raster_reference(self):
        raster = array_to_raster(
            arr_100,
            reference_raster=raster_100m,
        )
        self.assertEqual(isinstance(raster, gdal.Dataset), True, "Should be a GDAL dataframe")

    ''' ****************************************
        clip_raster
    **************************************** '''
    def test_clip_raster_no_reference(self):
        with self.assertRaises(AttributeError):
            clip_raster(raster_100m)

    def test_clip_raster_raster_reference(self):
        raster = clip_raster(raster_100m, reference_raster=raster_200m, quiet=True)
        self.assertEqual(isinstance(raster, gdal.Dataset), True, "Should be a GDAL dataframe")

    def test_clip_raster_raster_cutline(self):
        raster = clip_raster(raster_100m, cutline=geom_aalborg, quiet=True)
        self.assertEqual(isinstance(raster, gdal.Dataset), True, "Should be a GDAL dataframe")

# unittest.main()

red_edge = concatenate_images([b5, b6, b7], './red_edge.tif')
pca = dimension_reduction(red_edge, './red_edge_PCA_ica.tif', options={
    'method': 'ica',
    'rescale.outmin': 0,
    'rescale.outmax': 1,
    'nbcomp': 1,
    'normalize': 'YES',
})
rescale(pca, './red_edge_PCA_scaled_ica.tif')
