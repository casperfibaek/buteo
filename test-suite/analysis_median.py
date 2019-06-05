import sys
import os
import numpy as np
from glob import glob

sys.path.append('../lib')
from array_to_raster import array_to_raster
from raster_to_array import raster_to_array
from clip_raster import clip_raster

base = 'D:\\PhD\\Projects\\Byggemodning\\slc\\'


# array_to_raster(np.maximum(asc, desc), reference_raster=ref_path, out_raster=os.path.join(base, '2019_db.tif'))

images = glob(f"{base}*.tif")
images_array = np.array(list(map(lambda x: raster_to_array(x), images)))

array_to_raster(np.min(images_array, axis=0), out_raster=os.path.join(base, 'inter-annual_coh.tif'), reference_raster=images[0])
# array_to_raster(np.median(images_array, axis=0), out_raster=os.path.join(base, '2018_coh_median.tif'), reference_raster=images[0])
# array_to_raster(np.average(images_array, axis=0), out_raster=os.path.join(base, '2018_coh_average.tif'), reference_raster=images[0])
