import sys; sys.path.append('..'); sys.path.append('../lib/')
import geopandas as gpd
import pandas as pd
import os
import shutil
from glob import glob
import subprocess
from multiprocessing import Pool, cpu_count
# from lib.orfeo_toolbox import execute_cli_function

from sen1mosaic.download import search, download, connectToAPI, decompress
from sen1mosaic.preprocess import processFiles
from sen1mosaic.mosaic import buildComposite

# from sen2mosaic.download import search, download, connectToAPI, decompress
# from mosaic_tool import mosaic_tile

# project_area = '../geometry/ghana_landarea.shp'
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

# api_connection = connectToAPI('casperfibaek', 'Goldfish12')

# data.reverse()


# for index, tile in enumerate(data):
#     # import pdb; pdb.set_trace()
#     sdf_grd = search(data_bounds[index], api_connection, start='20200315', end='20200401', producttype='GRD')
#     download(sdf_grd, api_connection, '/home/cfi/data')
#     sdf_slc = search(data_bounds[index], api_connection, start='20200315', end='20200401', producttype='SLC')
#     download(sdf_slc, api_connection, '/home/cfi/data')

s1_images = glob('/home/cfi/data/*GRDH*.*')
tmp_folder = '/home/cfi/tmp/'
dst_folder = '/home/cfi/mosaic/'

processFiles(s1_images, dst_folder, tmp_folder, verbose=True)
# buildComposite(['/home/cfi/mosaic/S1_processed_20200321_183318_181855_020794_0276DC.dim'], 'VV', dst_folder)


# for tile in data:
#     cloud_search = 5
#     min_images = 10
#     downloaded = False
#     sdf = None

#     while cloud_search <= 100:
#         sdf = search(tile, level='2A', start='20190101', end='20190401', maxcloud=cloud_search, minsize=100.0)
        
#         if len(sdf) >= min_images:
#             print(f"Found {len(sdf)} images of {tile} @ {cloud_search}% cloud cover - downloading.." )
#             download(sdf, '/home/cfi/data')
#             downloaded = True
#             break

#         print(f'Found too few images of {tile} @ {cloud_search}% cloud cover - skipping..')
#         cloud_search += 5
    
#     if downloaded == False:
#         download(sdf, '/home/cfi/data')
        

# wkt_proj = project_geom.crs.to_wkt()

# def calc_tile(tile):
#     tmp_dir = '/home/cfi/tmp/'
#     dst_dir = '/home/cfi/mosaic/'
#     images = glob(f'/home/cfi/data/*{tile}*.zip')
#     decompress(images, tmp_dir)
#     images = glob(f'{tmp_dir}*{tile}*')

#     mosaic_tile(
#         images,
#         dst_dir,
#         tile,
#         dst_projection=wkt_proj,
#     )

#     delete_files = glob(f"{tmp_dir}*{tile}*.*")
#     for f in delete_files:
#         try:
#             shutil.rmtree(f)
#         except:
#             pass

# for tile in data:
#     calc_tile(tile)

# src_dir = '/home/cfi/mosaic/'
# tmp_dir = '/home/cfi/tmp/'
# dst_dir = '/home/cfi/mosaic/merged/'

# call = ''

# TODO: Figure out relative calls..
# for band in ['B02', 'B03', 'B04', 'B08']:
#     images = glob(f"{src_dir}{band}*.tif")
#     images_str = " ".join(images)

#     cli = f'otbcli_Mosaic -il {images_str} -tmpdir {tmp_dir} -comp.feather large -harmo.method band -out "{dst_dir}{band}_mosaic.tif?&gdal:co:COMPRESS=DEFLATE&gdal:co:PREDICTOR=2&gdal:co:NUM_THREADS=ALL_CPUS&gdal:co:BIGTIFF=YES" uint16 &'
#     import pdb; pdb.set_trace()
    # subprocess.Popen(cli)
    
# b2 = 'otbcli_Mosaic -il /home/cfi/mosaic/B02_30NZN.tif /home/cfi/mosaic/B02_31NBJ.tif /home/cfi/mosaic/B02_30NVL.tif /home/cfi/mosaic/B02_31PBK.tif /home/cfi/mosaic/B02_30NZP.tif /home/cfi/mosaic/B02_30NXP.tif /home/cfi/mosaic/B02_31PBL.tif /home/cfi/mosaic/B02_30PXS.tif /home/cfi/mosaic/B02_30PXT.tif /home/cfi/mosaic/B02_30PZR.tif /home/cfi/mosaic/B02_30NVN.tif /home/cfi/mosaic/B02_30PXR.tif /home/cfi/mosaic/B02_30NXM.tif /home/cfi/mosaic/B02_30PVS.tif /home/cfi/mosaic/B02_30NWP.tif /home/cfi/mosaic/B02_30NYN.tif /home/cfi/mosaic/B02_30PYR.tif /home/cfi/mosaic/B02_30NWL.tif /home/cfi/mosaic/B02_30PWT.tif /home/cfi/mosaic/B02_31NBG.tif /home/cfi/mosaic/B02_30PYS.tif /home/cfi/mosaic/B02_30NZM.tif /home/cfi/mosaic/B02_30NWN.tif /home/cfi/mosaic/B02_30NYP.tif /home/cfi/mosaic/B02_30PWQ.tif /home/cfi/mosaic/B02_30NVP.tif /home/cfi/mosaic/B02_30NYM.tif /home/cfi/mosaic/B02_30PYT.tif /home/cfi/mosaic/B02_30NYL.tif /home/cfi/mosaic/B02_30NVM.tif /home/cfi/mosaic/B02_30PWR.tif /home/cfi/mosaic/B02_30PZQ.tif /home/cfi/mosaic/B02_30PXQ.tif /home/cfi/mosaic/B02_31NBH.tif /home/cfi/mosaic/B02_30NXN.tif /home/cfi/mosaic/B02_30NXL.tif /home/cfi/mosaic/B02_30NWM.tif /home/cfi/mosaic/B02_30PZS.tif /home/cfi/mosaic/B02_30PZT.tif /home/cfi/mosaic/B02_31PBM.tif /home/cfi/mosaic/B02_30PYQ.tif /home/cfi/mosaic/B02_30PWS.tif -tmpdir /home/cfi/tmp/ -comp.feather large -harmo.method band -out "/home/cfi/mosaic/merged/B02_mosaic.tif?&gdal:co:COMPRESS=DEFLATE&gdal:co:PREDICTOR=2&gdal:co:NUM_THREADS=ALL_CPUS&gdal:co:BIGTIFF=YES" uint16'
# b3 = 'otbcli_Mosaic -il /home/cfi/mosaic/B03_31NBG.tif /home/cfi/mosaic/B03_30PYS.tif /home/cfi/mosaic/B03_30NXP.tif /home/cfi/mosaic/B03_30NWL.tif /home/cfi/mosaic/B03_30PVS.tif /home/cfi/mosaic/B03_30NYN.tif /home/cfi/mosaic/B03_30PZT.tif /home/cfi/mosaic/B03_30PXS.tif /home/cfi/mosaic/B03_30NYL.tif /home/cfi/mosaic/B03_30PWR.tif /home/cfi/mosaic/B03_30PWT.tif /home/cfi/mosaic/B03_30NZM.tif /home/cfi/mosaic/B03_31NBJ.tif /home/cfi/mosaic/B03_30NWP.tif /home/cfi/mosaic/B03_30NVL.tif /home/cfi/mosaic/B03_30NWN.tif /home/cfi/mosaic/B03_30NVM.tif /home/cfi/mosaic/B03_30PZS.tif /home/cfi/mosaic/B03_30NZN.tif /home/cfi/mosaic/B03_31PBM.tif /home/cfi/mosaic/B03_30PZQ.tif /home/cfi/mosaic/B03_30PXT.tif /home/cfi/mosaic/B03_30NXN.tif /home/cfi/mosaic/B03_31NBH.tif /home/cfi/mosaic/B03_30NWM.tif /home/cfi/mosaic/B03_30NXM.tif /home/cfi/mosaic/B03_30NVP.tif /home/cfi/mosaic/B03_30PXQ.tif /home/cfi/mosaic/B03_30PZR.tif /home/cfi/mosaic/B03_30NXL.tif /home/cfi/mosaic/B03_30PWS.tif /home/cfi/mosaic/B03_30PYR.tif /home/cfi/mosaic/B03_30NZP.tif /home/cfi/mosaic/B03_31PBL.tif /home/cfi/mosaic/B03_30NVN.tif /home/cfi/mosaic/B03_31PBK.tif /home/cfi/mosaic/B03_30PYT.tif /home/cfi/mosaic/B03_30NYM.tif /home/cfi/mosaic/B03_30PXR.tif /home/cfi/mosaic/B03_30NYP.tif /home/cfi/mosaic/B03_30PYQ.tif /home/cfi/mosaic/B03_30PWQ.tif -tmpdir /home/cfi/tmp/ -comp.feather large -harmo.method band -out "/home/cfi/mosaic/merged/B03_mosaic.tif?&gdal:co:COMPRESS=DEFLATE&gdal:co:PREDICTOR=2&gdal:co:NUM_THREADS=ALL_CPUS&gdal:co:BIGTIFF=YES" uint16'
# b4 = 'otbcli_Mosaic -il /home/cfi/mosaic/B04_30PYT.tif /home/cfi/mosaic/B04_30PXS.tif /home/cfi/mosaic/B04_30NVN.tif /home/cfi/mosaic/B04_30NZM.tif /home/cfi/mosaic/B04_31PBM.tif /home/cfi/mosaic/B04_30NVL.tif /home/cfi/mosaic/B04_30NXL.tif /home/cfi/mosaic/B04_30NVP.tif /home/cfi/mosaic/B04_30NWN.tif /home/cfi/mosaic/B04_30PZS.tif /home/cfi/mosaic/B04_30PWS.tif /home/cfi/mosaic/B04_30PYS.tif /home/cfi/mosaic/B04_30NWP.tif /home/cfi/mosaic/B04_30NXP.tif /home/cfi/mosaic/B04_30PZQ.tif /home/cfi/mosaic/B04_30NWM.tif /home/cfi/mosaic/B04_30NYN.tif /home/cfi/mosaic/B04_31NBH.tif /home/cfi/mosaic/B04_30PWQ.tif /home/cfi/mosaic/B04_30PXR.tif /home/cfi/mosaic/B04_31NBJ.tif /home/cfi/mosaic/B04_30PZR.tif /home/cfi/mosaic/B04_30PWT.tif /home/cfi/mosaic/B04_30NXN.tif /home/cfi/mosaic/B04_30NYL.tif /home/cfi/mosaic/B04_30NYP.tif /home/cfi/mosaic/B04_30NVM.tif /home/cfi/mosaic/B04_30PZT.tif /home/cfi/mosaic/B04_30PYR.tif /home/cfi/mosaic/B04_30PYQ.tif /home/cfi/mosaic/B04_30NWL.tif /home/cfi/mosaic/B04_30NYM.tif /home/cfi/mosaic/B04_31NBG.tif /home/cfi/mosaic/B04_30PVS.tif /home/cfi/mosaic/B04_31PBK.tif /home/cfi/mosaic/B04_31PBL.tif /home/cfi/mosaic/B04_30NZP.tif /home/cfi/mosaic/B04_30NXM.tif /home/cfi/mosaic/B04_30PXQ.tif /home/cfi/mosaic/B04_30NZN.tif /home/cfi/mosaic/B04_30PXT.tif /home/cfi/mosaic/B04_30PWR.tif -tmpdir /home/cfi/tmp/ -comp.feather large -harmo.method band -out "/home/cfi/mosaic/merged/B04_mosaic.tif?&gdal:co:COMPRESS=DEFLATE&gdal:co:PREDICTOR=2&gdal:co:NUM_THREADS=ALL_CPUS&gdal:co:BIGTIFF=YES" uint16'
# b8 = 'otbcli_Mosaic -il /home/cfi/mosaic/B08_31NBH.tif /home/cfi/mosaic/B08_30NXN.tif /home/cfi/mosaic/B08_30PYT.tif /home/cfi/mosaic/B08_30PXR.tif /home/cfi/mosaic/B08_31PBM.tif /home/cfi/mosaic/B08_30NZP.tif /home/cfi/mosaic/B08_30PYQ.tif /home/cfi/mosaic/B08_31NBG.tif /home/cfi/mosaic/B08_30PZQ.tif /home/cfi/mosaic/B08_30PYS.tif /home/cfi/mosaic/B08_30NYN.tif /home/cfi/mosaic/B08_30NWN.tif /home/cfi/mosaic/B08_30NWL.tif /home/cfi/mosaic/B08_30PWR.tif /home/cfi/mosaic/B08_30PWT.tif /home/cfi/mosaic/B08_30NYL.tif /home/cfi/mosaic/B08_30PWQ.tif /home/cfi/mosaic/B08_31PBL.tif /home/cfi/mosaic/B08_30NVN.tif /home/cfi/mosaic/B08_30NXL.tif /home/cfi/mosaic/B08_31NBJ.tif /home/cfi/mosaic/B08_30PXS.tif /home/cfi/mosaic/B08_30NWP.tif /home/cfi/mosaic/B08_30PZR.tif /home/cfi/mosaic/B08_30NVP.tif /home/cfi/mosaic/B08_30PZT.tif /home/cfi/mosaic/B08_30NZN.tif /home/cfi/mosaic/B08_30NVM.tif /home/cfi/mosaic/B08_30NWM.tif /home/cfi/mosaic/B08_31PBK.tif /home/cfi/mosaic/B08_30NVL.tif /home/cfi/mosaic/B08_30PYR.tif /home/cfi/mosaic/B08_30NYM.tif /home/cfi/mosaic/B08_30NXP.tif /home/cfi/mosaic/B08_30PVS.tif /home/cfi/mosaic/B08_30PZS.tif /home/cfi/mosaic/B08_30PWS.tif /home/cfi/mosaic/B08_30PXQ.tif /home/cfi/mosaic/B08_30NYP.tif /home/cfi/mosaic/B08_30PXT.tif /home/cfi/mosaic/B08_30NZM.tif /home/cfi/mosaic/B08_30NXM.tif -tmpdir /home/cfi/tmp/ -comp.feather large -harmo.method band -out "/home/cfi/mosaic/merged/B08_mosaic.tif?&gdal:co:COMPRESS=DEFLATE&gdal:co:PREDICTOR=2&gdal:co:NUM_THREADS=ALL_CPUS&gdal:co:BIGTIFF=YES" uint16'
# calc_tile(data[0])
# exit()
# pool = Pool(cpu_count())
# pool.map(calc_tile, data)
# pool.close()
# pool.join()


    
    # '30NVM', '30NVN', '30NVP', '30NWL', '30NVM', '30NWN', '30NWM' '30NWP'
    # if tile in ['30NVL', '30NVM', '30NVN', '30NVP', '30NWL', '30NVM', '30NWN', '30NWM' '30NWP', '30NXL', '30NXM', '30NXN', '30NXP', '30NYL', '30NYM', '30NYN', '30NYP', '30NZM', '30NZN', '30NZP', '30PVS', '30PWQ']:
    #     continue
    
    # if tile not in ['30NYM']:
    #     continue
    



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
