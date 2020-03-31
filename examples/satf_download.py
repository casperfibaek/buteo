import sys; sys.path.append('..'); sys.path.append('../lib/')
import geopandas as gpd
import pandas as pd
import os
import shutil
from glob import glob

from sen2mosaic.download import search, download, connectToAPI, decompress
from mosaic_tool import mosaic_tile

# project_area = '/mnt/c/Users/caspe/Desktop/Analysis/Data/vector/ghana_landarea.shp'
# project_geom = gpd.read_file(project_area)
# project_geom_wgs = project_geom.to_crs('EPSG:4326')

# tiles = '../geometry/sentinel2_tiles_world.shp'
# tiles_geom = gpd.read_file(tiles)
# tiles_dest = tiles_geom.to_crs(project_geom.crs)

# data = []
# data_bounds = []
# for index_g, geom in project_geom_wgs.iterrows():
#     for index_t, tile in tiles_geom.iterrows():
#         if geom['geometry'].intersects(tile['geometry']):
#             data.append(tile['Name'])
#             data_bounds.append(list(tile['geometry'].bounds))

# import pdb; pdb.set_trace()
# connectToAPI('test', 'test')

# data.reverse()

# extra_tiles = ['30NVL', '30NWL', '30NYN', '30NVM', '31PBM', '30NWN']

# for tile in data:
#     if tile in extra_tiles:
#         sdf = search(tile, level='2A', start='20191201', maxcloud=50, minsize=100.0)
#         download(sdf, '/mnt/d/data/')


tmp_dir = '/mnt/c/Users/caspe/Desktop/tmp/'
dst_dir = '/mnt/c/Users/caspe/Desktop/test/'

for index, tile in enumerate(['30NYL']):

    if len(glob(f"{dst_dir}*tile*")) != 0:
        continue
    
    # if tile in ['31NBG', '31NBH', '31NBJ', '31PBK', '31PBL', '31PBM']:
    if tile in ['30NYL']:
        # images = glob(f'/mnt/d/data/*{tile}*.zip')
        # decompress(images, tmp_dir)
        images = glob(f'{tmp_dir}*{tile}*')

        mosaic_tile(
            images,
            dst_dir,
            tile,
            # dst_projection=project_geom.crs.to_wkt(),
            feather=True,
            cutoff_invalid=2,
            cutoff_b1_cloud=800,
            cutoff_b1_ratio=0.80,
            invalid_contract=5,
            invalid_expand=61,
            border_dist=161,
            feather_dist=31,
            filter_tracking=True,
            filter_tracking_dist=9,
            filter_tracking_iterations=2,
            match_mean=True,
            match_quintile=0.25,
            max_days=30,
            max_images_include=10,
            max_search_images=15,
        )
        
        # delete_files = glob(f"{tmp_dir}*.*")
        # for f in delete_files:
        #     try:
        #         shutil.rmtree(f)
        #     except:
        #         pass


# import numpy as np
# import sys; sys.path.append('..'); sys.path.append('../lib/')
# from lib.raster_io import raster_to_array, array_to_raster

# # Pre-scaling of the mosaics.
# in_scaled = '/mnt/c/Users/caspe/Desktop/mosaic/'
# out_scaled = '/mnt/c/Users/caspe/Desktop/scaled/'
# scales = {
#     'B02': {
#         'vegetation': [],
#         'vegetation_q25': None,
#         'non_vegetation': [],
#         'non_vegetation_q25': None,
#     },
#     'B03': {
#         'vegetation': [],
#         'vegetation_q25': None,
#         'non_vegetation': [],
#         'non_vegetation_q25': None,
#     },
#     'B04': {
#         'vegetation': [],
#         'vegetation_q25': None,
#         'non_vegetation': [],
#         'non_vegetation_q25': None,
#     },
#     'B08': {
#         'vegetation': [],
#         'vegetation_q25': None,
#         'non_vegetation': [],
#         'non_vegetation_q25': None,
#     }
# }

# stats = {
#     'tile': {
#         'band': {
#             'vegetation_mean': None,
#             'non_vegetation_mean': None,
#             'scaling': 1,
#         },
#     },
# }

# match_quintile = 0.25

# for tile in data:

#     if tile not in ['31NBG', '31NBH', '31NBJ', '31PBK', '31PBL', '31PBM']:
#         continue

#     print(f'Calculating means for: {tile}')

#     stats[tile] = {}

#     # Since we are doing reprojection, the slc file can have nodata. Fill with zero
#     slc = raster_to_array(glob(in_scaled + f'slc_{tile}.tif')[0])
#     slc.fill_value = 0
#     slc = slc.filled()
    
#     vegetation_mask = np.ma.masked_equal(slc != 4, slc).data
#     vegetation_sum = vegetation_mask.size - vegetation_mask.sum()
    
#     if vegetation_sum == 0:
#         continue
    
#     non_vegetation_mask = np.ma.masked_equal(slc != 5, slc).data
#     non_vegetation_sum = non_vegetation_mask.size - non_vegetation_mask.sum()
    
#     if non_vegetation_sum == 0:
#         continue
    
#     for band in ['B02', 'B03', 'B04', 'B08']:
        
#         tile_path = glob(in_scaled + f'{band}_{tile}.tif')[0]
        
#         stats[tile][band] = {
#             'vegetation_mean': None,
#             'non_vegetation_mean': None,
#             'scaling': 1,
#         }
        
#         raw_array = raster_to_array(tile_path)
        
#         vegetation_mean = np.ma.array(raw_array, mask=vegetation_mask).mean()
#         non_vegetation_mean = np.ma.array(raw_array, mask=non_vegetation_mask).mean()

#         stats[tile][band]['vegetation_mean'] = vegetation_mean
#         scales[band]['vegetation'].append(vegetation_mean)

#         stats[tile][band]['non_vegetation_mean'] = non_vegetation_mean
#         scales[band]['non_vegetation'].append(non_vegetation_mean)

# for band in ['B02', 'B03', 'B04', 'B08']:
#     scales[band]['vegetation_q25'] = np.quantile(scales[band]['vegetation'] ,match_quintile)
#     scales[band]['non_vegetation_q25'] = np.quantile(scales[band]['non_vegetation'], match_quintile)

# for tile in data:

#     if tile not in ['31NBG', '31NBH', '31NBJ', '31PBK', '31PBL', '31PBM']:
#         continue

#     print(f'Scaling: {tile}')
#     for band in ['B02', 'B03', 'B04', 'B08']:
#         tile_path = glob(in_scaled + f'{band}_{tile}.tif')[0]
#         stats[tile][band]['scaling'] = (
#               (scales[band]['vegetation_q25'] / stats[tile][band]['vegetation_mean'])
#             + (scales[band]['non_vegetation_q25'] / stats[tile][band]['non_vegetation_mean'])
#             / 2)

#         input_band = np.ma.array(raster_to_array(tile_path))
#         input_band.fill_value = 65535

#         array_to_raster(
#             np.ma.multiply(input_band, stats[tile][band]['scaling']).astype('uint16'),
#             reference_raster=tile_path,
#             out_raster=out_scaled + f'{band}_{tile}_scaled.tif',
#         )
