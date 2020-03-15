import sys; sys.path.append('..')
from lib.raster_clip import clip_raster
from glob import glob
from pathlib import Path


folder = '/mnt/c/users/caspe/desktop/Analysis/data/'
master = folder + 'clipped/reference.tif'
cutline = folder + 'vector/project_area.shp'

layers = glob(folder + 'nightlights.tif')

for layer in layers:
    outname = folder + 'clipped/' + Path(layer).stem + '.tif'
    clip_raster(layer, reference_raster=master, cutline=cutline, out_raster=outname, scale_to_reference=True, quiet=True)
        
