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

# connectToAPI('test', 'test')

# data.reverse()

# for tile in data:
#     sdf = search(tile, level='2A', start='20200101', maxcloud=50, minsize=50.0)
#     download(sdf, '/mnt/d/data/')

tmp_dir = '/mnt/c/Users/caspe/Desktop/tmp/'
dst_dir = '/mnt/d/mosaic/'

for index, tile in enumerate(data):

    if len(glob(f"{dst_dir}*tile*")) != 0:
        continue

    images = glob(f'/mnt/d/data/*{tile}*')
    decompress(images, tmp_dir)
    images = glob(f'{tmp_dir}*{tile}*')
    images.reverse()
    
    mosaic_tile(images, dst_dir, tile)
    
    delete_files = glob(f"{tmp_dir}*.*")
    for f in delete_files:
        try:
            shutil.rmtree(f)
        except:
            pass
