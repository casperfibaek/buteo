import sys; sys.path.append('..'); sys.path.append('../lib/')
import geopandas as gpd
import pandas as pd
import os
import shutil
from glob import glob

from sen2mosaic.download import search, download, connectToAPI, decompress
from sen2mosaic.mosaic import build_mosaic

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
    
    if tile != '30NZN':
        continue

    images = glob(f'/mnt/d/data/*{tile}*')
    decompress(images, tmp_dir)
    images = glob(f'{tmp_dir}*{tile}*')
    epsg = project_geom.crs.to_epsg()
    bounds = list(tiles_dest.loc[tiles_dest['Name'] == tile]['geometry'].iloc[0].bounds)
    for i, x in enumerate(bounds):
        bounds[i] = int(round(x))  # Only appropriate because I know its sentinel 2 imagery.
    
    build_mosaic(images, bounds, epsg, resolution=60, level='2A', verbose=True, percentile=15.0, processes=8, improve_mask=False, colour_balance=False, output_name=f'{tile}', step=2000,  output_dir=dst_dir, temp_dir=tmp_dir)
    
    delete_files = glob(f"{tmp_dir}*.*")
    for f in delete_files:
        try:
            shutil.rmtree(f)
        except:
            pass
