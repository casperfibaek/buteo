import sys; sys.path.append('..'); sys.path.append('../lib/')
from lib.raster_io import raster_to_array, array_to_raster
from lib.raster_reproject import reproject
import xml.etree.ElementTree as ET
from glob import glob
from datetime import datetime
from osgeo import ogr
import os

from sen1mosaic.preprocess import processFiles_coherence
# from sen1mosaic.mosaic import buildComposite

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


def get_metadata(image_paths):
    images_obj = []

    for img in image_paths:
        kml = f"{img}/preview/map-overlay.kml"

        timestr = str(img.rsplit('.')[0].split('/')[-1].split('_')[5])
        timestamp = datetime.strptime(timestr, "%Y%m%dT%H%M%S").timestamp()

        meta = {
            'path': img,
            'timestamp': timestamp,
            'footprint_wkt': s1_kml_to_bbox(kml),
        }

        images_obj.append(meta)

    return images_obj


def coherence_step1(image_paths, out_folder, gpt="~/esa_snap/bin/gpt"):
    images_obj = get_metadata(image_paths)
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
            processFiles_coherence(metadata['path'], images_obj[best_overlap]['path'], out_folder + str(int(metadata['timestamp'])) + '_step1', step=1, gpt=gpt)

            processed.append([best_overlap, index_i])

        print(processed)


def coherence_step2(input_folder, gpt="~/esa_snap/bin/gpt"):
    step1_images = glob(input_folder + '*step1*.dim')
    completed = 0
    for image in step1_images:
        outname = image.rsplit('_', 1)[0] + '_step2'
        try:
            processFiles_coherence(image, None, outname, step=2, gpt=gpt)
        except:
            print('Failed to processes: ', outname)
        completed += 1
        print(str(completed) + '/' + str(len(step1_images)))


if __name__ == "__main__":
    # base_folder = "/home/cfi/Desktop/sentinel1_midtjylland/descending/slc/"
    # out_folder = "/home/cfi/Desktop/sentinel1_midtjylland/descending/slc_processed/"
    # slc_files = glob(base_folder + "*.SAFE")
    
    # coherence_step1(slc_files, out_folder, gpt="~/esa_snap/bin/gpt")

    ascending_folder = "/home/cfi/Desktop/sentinel1_midtjylland/ascending/slc_processed/"
    descending_folder = "/home/cfi/Desktop/sentinel1_midtjylland/descending/slc_processed/"
    # coherence_step2(ascending_folder)
    # coherence_step2(descending_folder)

    proj = 'PROJCS["ETRS89 / UTM zone 32N",GEOGCS["ETRS89",DATUM["European_Terrestrial_Reference_System_1989",SPHEROID["GRS 1980",6378137,298.257222101,AUTHORITY["EPSG","7019"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6258"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4258"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",9],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","25832"]]'

    # ascending_images = glob(ascending_folder + "*step2*/*.img")
    # for img in ascending_images:
    #     name = os.path.basename(img).split(".", 1)[0] + "_asc.tif"
    #     reproject(array_to_raster(raster_to_array(img), None, img), ascending_folder + name, target_projection=proj)

    descdending_images = glob(descending_folder + "*step2*/*.img")
    for img in descdending_images:
        name = os.path.basename(img).split(".", 1)[0] + "_desc.tif"
        reproject(array_to_raster(raster_to_array(img), None, img), descending_folder + name, target_projection=proj)
