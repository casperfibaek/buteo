import sys; sys.path.append('..'); sys.path.append('../lib/')
import geopandas as gpd
import pandas as pd
import os
import shutil
from glob import glob

from sen2mosaic.download import search, download, connectToAPI, decompress
from mosaic_tool import mosaic_tile

project_area = '/mnt/c/Users/caspe/Desktop/Analysis/Data/vector/ghana_landarea.shp'
project_geom = gpd.read_file(project_area)
project_geom_wgs = project_geom.to_crs('EPSG:4326')

tiles = '../geometry/sentinel2_tiles_world.shp'
tiles_geom = gpd.read_file(tiles)
tiles_dest = tiles_geom.to_crs(project_geom.crs)

data = []
data_bounds = []
for index_g, geom in project_geom_wgs.iterrows():
    for index_t, tile in tiles_geom.iterrows():
        if geom['geometry'].intersects(tile['geometry']):
            data.append(tile['Name'])
            data_bounds.append(list(tile['geometry'].bounds))

# import pdb; pdb.set_trace()
# connectToAPI('test', 'test')

# data.reverse()

# extra_tiles = ['30NVL', '30NWL', '30NYN', '30NVM', '31PBM', '30NWN']

# for tile in data:
#     if tile in extra_tiles:
#         sdf = search(tile, level='2A', start='20191201', maxcloud=50, minsize=50.0)
#         download(sdf, '/mnt/d/data/')

# TODO: Mean match using SLC pre-orfeo.

tmp_dir = '/mnt/c/Users/caspe/Desktop/tmp/'
dst_dir = '/mnt/d/mosaic/'

for index, tile in enumerate(data):

    if len(glob(f"{dst_dir}*tile*")) != 0:
        continue

    images = glob(f'/mnt/d/data/*{tile}*.zip')
    decompress(images, tmp_dir)
    images = glob(f'{tmp_dir}*{tile}*')
    images.reverse()
    
    mosaic_tile(images, dst_dir, tile, project_geom.crs.to_proj4())
    
    delete_files = glob(f"{tmp_dir}*.*")
    for f in delete_files:
        try:
            shutil.rmtree(f)
        except:
            pass
