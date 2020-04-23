import sys; sys.path.append('..'); sys.path.append('../lib/')
import geopandas as gpd
import pandas as pd
import os
import shutil
import subprocess
from glob import glob
from osgeo import ogr
from multiprocessing import Pool, cpu_count
from shutil import copyfile
# from lib.orfeo_toolbox import execute_cli_function
from lib.raster_reproject import reproject
from lib.raster_clip import clip_raster
from lib.raster_io import raster_to_metadata

from sen1mosaic.download import search, download, connectToAPI, decompress
from sen1mosaic.preprocess import processFiles, processFiles_coherence
from sen1mosaic.mosaic import buildComposite
from pyproj import CRS

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
tmp_folder = '/home/cfi/tmp/distances/'
dst_folder = '/home/cfi/mosaic/merged/'


import xml.etree.ElementTree as ET
from datetime import datetime
from shapely.geometry import shape
import zipfile
import json
# s1_images = glob('/home/cfi/tmp/*.dim')
# s1_images = glob('/home/cfi/data/slc/*/manifest.safe')

# images_obj = []

# def s1_kml_to_bbox(path_to_kml):
#     root = ET.parse(path_to_kml).getroot()
#     for elem in root.iter():
#         if elem.tag == 'coordinates':
#             coords = elem.text
#             break

#     coords = coords.split(',')
#     coords[0] = coords[-1] + ' ' + coords[0]
#     del coords[-1]
#     coords.append(coords[0])

#     min_x = 180
#     max_x = -180
#     min_y = 90
#     max_y = -90

#     for i in range(len(coords)):
#         intermediate = coords[i].split(' ')
#         intermediate.reverse()

#         intermediate[0] = float(intermediate[0])
#         intermediate[1] = float(intermediate[1])

#         if intermediate[0] < min_x:
#             min_x = intermediate[0]
#         elif intermediate[0] > max_x:
#             max_x = intermediate[0]
        
#         if intermediate[1] < min_y:
#             min_y = intermediate[1]
#         elif intermediate[1] > max_y:
#             max_y = intermediate[1]

#     footprint = f"POLYGON (({min_x} {min_y}, {min_x} {max_y}, {max_x} {max_y}, {max_x} {min_y}, {min_x} {min_y}))"
    
#     return footprint

# for img in s1_images:
#     folder = img.rsplit('/', 1)[0]
#     kml = f"{folder}/preview/map-overlay.kml"

#     timestr = str(img.rsplit('.')[0].split('/')[-1].split('_')[5])
#     timestamp = datetime.strptime(timestr, "%Y%m%dT%H%M%S").timestamp()

#     meta = {
#         'path': img,
#         'timestamp': timestamp,
#         'footprint_wkt': s1_kml_to_bbox(kml),
#     }

#     images_obj.append(meta)

# processed = []
# for index_i, metadata in enumerate(images_obj):


#     # Find the image with the largest intersection
#     footprint = ogr.CreateGeometryFromWkt(metadata['footprint_wkt'])
#     highest_area = 0
#     best_overlap = False

#     for index_j in range(len(images_obj)):
#         if index_j == index_i: continue
        
#         comp_footprint = ogr.CreateGeometryFromWkt(images_obj[index_j]['footprint_wkt'])

#         intersection = footprint.Intersection(comp_footprint)

#         if intersection == None: continue
        
#         area = intersection.Area()
#         if area > highest_area:
#             highest_area = area
#             best_overlap = index_j

#     skip = False
#     if [index_i, best_overlap] in processed:
#         skip = True

#     if best_overlap is not False and skip is False:
#         processFiles_coherence(metadata['path'], images_obj[best_overlap]['path'], dst_folder + str(int(metadata['timestamp'])) + '_step1', step=1)

#         processed.append([best_overlap, index_i])
#     # .rsplit('/', 1)[0]

#     print(processed)

# step1_images = glob(dst_folder + '*step1*.dim')
# completed = 0
# for image in step1_images:
#     outname = image.rsplit('_', 1)[0] + '_step2'
#     try:
#         processFiles_coherence(image, None, outname, step=2)
#     except:
#         print('Failed to processes: ', outname)
#     completed += 1
#     print(str(completed) + '/' + str(len(step1_images)))


coh = glob('/home/cfi/mosaic/merged/*step2*.data/*.img')
# coh_str = ' '.join(coh)
# cli = f'otbcli_Mosaic -il {coh_str} -tmpdir {tmp_folder} -nodata 0 -comp.feather large -out "{dst_folder}gamma0_VV_coh_mosaic_feathered.tif?&gdal:co:COMPRESS=DEFLATE&gdal:co:PREDICTOR=2&gdal:co:NUM_THREADS=ALL_CPUS&gdal:co:BIGTIFF=YES" float'

# import pdb; pdb.set_trace()

# completed = 0
# for image in coh:
#     name = image.rsplit('/', 1)[1].split('.')[0] + '_' + str(completed) + '.tif'

#     reproject(image, dst_folder + 'reprojected/' + name, target_projection=CRS.from_string("epsg:32630"))
#     completed += 1

#     print(f'Completed: {completed}/{len(coh)}')


coh = glob('/home/cfi/mosaic/merged/reprojected/*.tif')
coh_str = ' '.join(coh)
cli = f'otbcli_Mosaic -il {coh_str} -tmpdir {tmp_folder} -nodata 0 -comp.feather large -out "{dst_folder}gamma0_VV_coh_mosaic_feathered.tif?&gdal:co:COMPRESS=DEFLATE&gdal:co:PREDICTOR=2&gdal:co:NUM_THREADS=ALL_CPUS&gdal:co:BIGTIFF=YES" float'
import pdb; pdb.set_trace()
# subprocess.Popen(comp)

# processFiles(s1_images, dst_folder, tmp_folder, verbose=True)
# buildComposite(s1_images, 'VV', dst_folder)

# s1_images_str = ' '.join(glob('/home/cfi/mosaic/*.tif'))
# cli = f'otbcli_Mosaic -il {s1_images_str} -tmpdir {tmp_folder} -nodata 0 -comp.feather large -out "{dst_folder}gamma0_VV_mosaic_feathered.tif?&gdal:co:COMPRESS=DEFLATE&gdal:co:PREDICTOR=2&gdal:co:NUM_THREADS=ALL_CPUS&gdal:co:BIGTIFF=YES" float'

# import pdb; pdb.set_trace()
# clip_raster(dst_folder + 'gamma0_VV_mosaic.tif', out_raster=dst_folder + 'gamma0_VV_mosaic_clip.tif', cutline='/home/cfi/yellow/geometry/ghana_5km_buffer.shp', cutline_all_touch=True)

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
    