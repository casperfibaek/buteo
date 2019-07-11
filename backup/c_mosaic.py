import numpy as np
from glob import glob

from raster_to_array import raster_to_array
from array_to_raster import array_to_raster
from resample import resample


def name_sort(img_path):
    filename = img_path.rsplit('\\', 1)[1]
    date_string = filename.split('_')[1].split('T')
    return int(''.join(date_string))

# Open SLC of newest file
# SLC_20m = raster_to_array('E:\\scratch_pad\\T33UUB_20180412T102021_SCL_20m.jp2')
# B02_20m = raster_to_array('E:\\scratch_pad\\T33UUB_20180412T102021_B02_20m.jp2')

# SLC_mask = np.isin(SLC_20m, [0, 1, 3, 8, 9, 10, 11])


def band_limit(band_name):
    if band_name is 'B04':
        return 2000
    else:
        return 9000

# B02_20m_masked = np.ma.masked_array(B02_20m, mask=SLC_mask)
# array_to_raster(B02_20m_masked, reference_raster='E:\\scratch_pad\\T33UUB_20180412T102021_B02_20m.jp2', out_raster='E:\\scratch_pad\\B02_masked.tif')

base = 'E:\\sentinel_2_data\\32\\VPH\\'
images = glob(f"{base}*_B*.jp2")
images.sort(key=name_sort)

base_image = raster_to_array(images[0])

for image in images:
    folder = image.rsplit('\\', 1)[0]
    filename = image.rsplit('\\', 1)[1]
    elements = filename.split('_')
    elements[2] = 'SCL'
    elements[3] = '20m.jp2'
    SCL_file = f"{folder}\\{'_'.join(elements)}"
    SLC_file = raster_to_array(resample(SCL_file, reference_raster=image))
    SLC_mask = np.logical_and((SLC_file < 1500), np.isin(SLC_file, [0, 1, 3, 8, 9, 10, 11], invert=True))

    img_tile = raster_to_array(image)
    np.copyto(base_image, img_tile, casting='no', where=SLC_mask)

array_to_raster(base_image, reference_raster=images[0], out_raster=f"{base}mosaic_v6.tif")
