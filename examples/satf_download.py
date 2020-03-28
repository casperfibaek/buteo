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
#         sdf = search(tile, level='2A', start='20191201', maxcloud=50, minsize=100.0)
#         download(sdf, '/mnt/d/data/')


tmp_dir = '/mnt/c/Users/caspe/Desktop/tmp/'
dst_dir = '/mnt/d/mosaic/'

for index, tile in enumerate(data):

    if len(glob(f"{dst_dir}*tile*")) != 0:
        continue

    images = glob(f'/mnt/d/data/*{tile}*.zip')
    decompress(images, tmp_dir)
    images = glob(f'{tmp_dir}*{tile}*')
    
    mosaic_tile(images, dst_dir, tile, project_geom.crs.to_proj4())
    
    delete_files = glob(f"{tmp_dir}*.*")
    for f in delete_files:
        try:
            shutil.rmtree(f)
        except:
            pass

# TODO: Mean match using SLC pre-orfeo.
# TODO: Figure out bug with 31NBG & 31NYL

import numpy as np
import sys; sys.path.append('..'); sys.path.append('../lib/')
from lib.raster_io import raster_to_array, array_to_raster

# Lets attempt a pre-scaling of the mosaics.
out_scaled = '/mnt/d/scaled/'

scales = {
    'B02': {
        'vegetation': [],
        'vegetation_q25': None,
        'non_vegetation': [],
        'non_vegetation_q25': None,
    },
    'B03': {
        'vegetation': [],
        'vegetation_q25': None,
        'non_vegetation': [],
        'non_vegetation_q25': None,
    },
    'B04': {
        'vegetation': [],
        'vegetation_q25': None,
        'non_vegetation': [],
        'non_vegetation_q25': None,
    },
    'B08': {
        'vegetation': [],
        'vegetation_q25': None,
        'non_vegetation': [],
        'non_vegetation_q25': None,
    }
}

stats = {
    'tile': {
        'band': {
            'vegetation_mean': None,
            'non_vegetation_mean': None,
            'scaling': 1,
        },
    },
}   

for tile in data:
    slc = glob(dst_dir + f'slc_{tile}.tif')[0]
    for band in ['B02', 'B03', 'B04', 'B08']:
        tile_path = glob(dst_dir, f'{band}_{tile}.tif')
        
        vegetation_mask = np.ma.masked_equal(slc != 4, slc).data
        vegetation_mean = np.ma.array(raster_to_array(tile_path), mask=vegetation_mask).mean()
        stats[tile]['vegetation_mean'] = vegetation_mean
        scales[band]['vegetation'].append(vegetation_mean)

        non_vegetation_mask = np.ma.masked_equal(slc != 5, slc).data
        non_vegetation_mean = np.ma.array(raster_to_array(tile_path), mask=vegetation_mask).mean()
        stats[tile]['non_vegetation_mean'] = vegetation_mean
        scales[band]['non_vegetation'].append(non_vegetation_mean)

for band in ['B02', 'B03', 'B04', 'B08']:
    scales[band]['vegetation_q25'] = np.quantile(scales[band]['vegetation'], 0.25)
    scales[band]['non_vegetation_q25'] = np.quantile(scales[band]['vegetation'], 0.25)

for tile in data:
    for band in ['B02', 'B03', 'B04', 'B08']:
        tile_path = glob(dst_dir, f'{band}_{tile}.tif')
        stats[tile][band]['scaling'] = (
            (scales[band]['vegetation_q25'] / stats[tile][band]['vegetation_mean'])
            + (scales[band]['non_vegetation_q25'] / stats[tile][band]['non_vegetation_mean'])
            / 2)

        array_to_raster(
            raster_to_array(tile_path) * stats[tile][band]['scaling'],
            reference_raster=tile_path,
            out_raster=out_scaled + f'{band}_{tile}_scaled.tif',
        )