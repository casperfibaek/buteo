import sys; sys.path.append('..'); sys.path.append('../lib/')
import geopandas as gpd
import pandas as pd
import os
import shutil
import subprocess
from glob import glob
from osgeo import ogr
from shutil import copyfile
from lib.raster_reproject import reproject
from lib.raster_clip import clip_raster
from pyproj import CRS

from sen1mosaic.download import search, download, connectToAPI, decompress
from sen1mosaic.preprocess import processFiles, processFiles_coherence
from sen1mosaic.mosaic import buildComposite

import xml.etree.ElementTree as ET
from datetime import datetime
from shapely.geometry import shape
import zipfile
import json
s1_images = glob('/home/cfi/tmp/*.dim')
s1_images = glob('/home/cfi/data/slc/*/manifest.safe')

images_obj = []

def s1_kml_to_bbox(path_to_kml):
    root = ET.parse(path_to_kml).getroot()
    for elem in root.iter():
        if elem.tag == 'coordinates':
            coords = elem.text
            break

    coords = coords.split(',')
    coords[0] = coords[-1] + ' ' + coords[0]
    del coords[-1]
    coords.append(coords[0])

    min_x = 180
    max_x = -180
    min_y = 90
    max_y = -90

    for i in range(len(coords)):
        intermediate = coords[i].split(' ')
        intermediate.reverse()

        intermediate[0] = float(intermediate[0])
        intermediate[1] = float(intermediate[1])

        if intermediate[0] < min_x:
            min_x = intermediate[0]
        elif intermediate[0] > max_x:
            max_x = intermediate[0]
        
        if intermediate[1] < min_y:
            min_y = intermediate[1]
        elif intermediate[1] > max_y:
            max_y = intermediate[1]

    footprint = f"POLYGON (({min_x} {min_y}, {min_x} {max_y}, {max_x} {max_y}, {max_x} {min_y}, {min_x} {min_y}))"
    
    return footprint

for img in s1_images:
    folder = img.rsplit('/', 1)[0]
    kml = f"{folder}/preview/map-overlay.kml"

    timestr = str(img.rsplit('.')[0].split('/')[-1].split('_')[5])
    timestamp = datetime.strptime(timestr, "%Y%m%dT%H%M%S").timestamp()

    meta = {
        'path': img,
        'timestamp': timestamp,
        'footprint_wkt': s1_kml_to_bbox(kml),
    }

    images_obj.append(meta)

processed = []
for index_i, metadata in enumerate(images_obj):


    # Find the image with the largest intersection
    footprint = ogr.CreateGeometryFromWkt(metadata['footprint_wkt'])
    highest_area = 0
    best_overlap = False

    for index_j in range(len(images_obj)):
        if index_j == index_i: continue
        
        comp_footprint = ogr.CreateGeometryFromWkt(images_obj[index_j]['footprint_wkt'])

        intersection = footprint.Intersection(comp_footprint)

        if intersection == None: continue
        
        area = intersection.Area()
        if area > highest_area:
            highest_area = area
            best_overlap = index_j

    skip = False
    if [index_i, best_overlap] in processed:
        skip = True

    if best_overlap is not False and skip is False:
        processFiles_coherence(metadata['path'], images_obj[best_overlap]['path'], dst_folder + str(int(metadata['timestamp'])) + '_step1', step=1)

        processed.append([best_overlap, index_i])
    # .rsplit('/', 1)[0]

    print(processed)

step1_images = glob(dst_folder + '*step1*.dim')
completed = 0
for image in step1_images:
    outname = image.rsplit('_', 1)[0] + '_step2'
    try:
        processFiles_coherence(image, None, outname, step=2)
    except:
        print('Failed to processes: ', outname)
    completed += 1
    print(str(completed) + '/' + str(len(step1_images)))


coh = glob('/home/cfi/mosaic/merged/*step2*.data/*.img')
coh_str = ' '.join(coh)
cli = f'otbcli_Mosaic -il {coh_str} -tmpdir {tmp_folder} -nodata 0 -comp.feather large -out "{dst_folder}gamma0_VV_coh_mosaic_feathered.tif?&gdal:co:COMPRESS=DEFLATE&gdal:co:PREDICTOR=2&gdal:co:NUM_THREADS=ALL_CPUS&gdal:co:BIGTIFF=YES" float'

import pdb; pdb.set_trace()

completed = 0
for image in coh:
    name = image.rsplit('/', 1)[1].split('.')[0] + '_' + str(completed) + '.tif'

    reproject(image, dst_folder + 'reprojected/' + name, target_projection=CRS.from_string("epsg:32630"))
    completed += 1

    print(f'Completed: {completed}/{len(coh)}')


coh = glob('/home/cfi/mosaic/merged/reprojected/*.tif')
coh_str = ' '.join(coh)
cli = f'otbcli_Mosaic -il {coh_str} -tmpdir {tmp_folder} -nodata 0 -comp.feather large -out "{dst_folder}gamma0_VV_coh_mosaic_feathered.tif?&gdal:co:COMPRESS=DEFLATE&gdal:co:PREDICTOR=2&gdal:co:NUM_THREADS=ALL_CPUS&gdal:co:BIGTIFF=YES" float'
import pdb; pdb.set_trace()
subprocess.Popen(comp)

processFiles(s1_images, dst_folder, tmp_folder, verbose=True)
buildComposite(s1_images, 'VV', dst_folder)

s1_images_str = ' '.join(glob('/home/cfi/mosaic/*.tif'))
cli = f'otbcli_Mosaic -il {s1_images_str} -tmpdir {tmp_folder} -nodata 0 -comp.feather large -out "{dst_folder}gamma0_VV_mosaic_feathered.tif?&gdal:co:COMPRESS=DEFLATE&gdal:co:PREDICTOR=2&gdal:co:NUM_THREADS=ALL_CPUS&gdal:co:BIGTIFF=YES" float'

import pdb; pdb.set_trace()
clip_raster(dst_folder + 'gamma0_VV_mosaic.tif', out_raster=dst_folder + 'gamma0_VV_mosaic_clip.tif', cutline='/home/cfi/yellow/geometry/ghana_5km_buffer.shp', cutline_all_touch=True)
