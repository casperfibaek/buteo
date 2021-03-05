
#%%
import sys; sys.path.append('..'); sys.path.append('../lib/')
from lib.raster_io import raster_to_array, array_to_raster
from lib.raster_clip import clip_raster
from lib.raster_reproject import reproject
# from lib.stats_filters import median_filter
from lib.stats_local import kernel_filter
from lib.stats_kernel import create_kernel
from glob import glob
import os
import numpy as np

from sen1mosaic.preprocess import correctionGraph

folder = "C:/Users/caspe/Desktop/noise_test/"
output_folder = folder + "processed/"
images = glob(folder + "*.zip")

#%%
for image in images:
    base = os.path.basename(image)
    basename = os.path.splitext(base)[0]
    # correctionGraph(image, f"{basename}_step1", output_folder, graph='backscatter_step1.xml')
    # correctionGraph(f"{output_folder}/{basename}_step1.dim", f"{basename}_step2", output_folder, graph='backscatter_step2.xml')

    vh_path = f"{output_folder}/{basename}_step2.data/Gamma0_VH.img"
    vv_path = f"{output_folder}/{basename}_step2.data/Gamma0_VV.img"

    array_to_raster(raster_to_array(vh_path), reference_raster=vh_path, out_raster=output_folder + basename + '_vh.tif')
    array_to_raster(raster_to_array(vv_path), reference_raster=vv_path, out_raster=output_folder + basename + '_vv.tif')

#%%
folder = "C:/Users/caspe/Desktop/noise_test/processed/aligned/"
vv_images = glob(folder + "*_vv_*.tif")
kernel_3d = create_kernel(5, sigma=1, dim=3, depth=3)
kernel_2d = create_kernel(5, sigma=1, dim=2, depth=3)
# %%
base = None
for index, image in enumerate(vv_images):
    arr = raster_to_array(image)
    med = kernel_filter(arr, kernel_2d, "median", 'float32')

    if index == 0:
        base = np.multiply(med, kernel_3d[index].sum())
    else:
        base = np.add(base, np.multiply(med, kernel_3d[index].sum()))

array_to_raster(base, out_raster=folder + "filtered_vv_median5.tif", reference_raster=image)
# %%

# %%
# The last step is to add median merge over multiple dimensions.