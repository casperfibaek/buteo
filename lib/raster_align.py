import sys; sys.path.append('..')
from lib.raster_clip import clip_raster
from glob import glob
from pathlib import Path


def align(input_rasters, target_projection=None, pixel_size=None)

folder = '/mnt/c/users/caspe/desktop/Analysis/data/'
master = folder + 'dem_slope.tif'
cutline = folder + 'vector/project_area.shp'

layers = glob(folder + 'nightlights.tif')

for layer in layers:
    outname = folder + Path(layer).stem + '_clip.tif'
    clip_raster(layer, reference_raster=master, cutline=cutline, out_raster=outname, scale_to_reference=True, quiet=False)
        
