import os
from osgeo import gdal, gdal_array
import numpy as np

FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/buteo/tests/"
img = os.path.join(FOLDER, "test_image_rgb_8bit.tif")

opened = gdal.Open(img)
band = opened.GetRasterBand(1)
gdal_data_type = band.DataType

numpy_data_type = np.dtype(gdal_array.GDALTypeCodeToNumericTypeCode(gdal_data_type))


gdal_array.NumericTypeCodeToGDALTypeCode("uint8")

import pdb; pdb.set_trace()