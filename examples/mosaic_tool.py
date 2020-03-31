import sys; sys.path.append('..'); sys.path.append('../lib/')
from lib.raster_io import raster_to_array, array_to_raster
from lib.raster_resample import resample
from lib.stats_filters import mode_filter, feather_s2_filter
from lib.stats_kernel import create_kernel
from lib.utils_core import weighted_quantile
import cv2
import os
import xml.etree.ElementTree as ET
import datetime
from glob import glob
import numpy as np


# INPUT: Array of sentinel 2 .SAFE files
# Verify input, verify all same tile, notify if .zip

# Read metadata of all files
## Get sensing time
## Get cloud coverage
## Get amount of valid pixels

# Select image with most cloudfree pixels
# Sort input array according to time from selected image
# Keep uint array with source pixels
# Iteratively check nearest images 3 for valid pixels
## Add min of valid pixels to source image (> 200 && < 3000)
## Use the band 4 as synth.
# Count invalid pixels
# If invalid pixels, search next 3 images for valid pixels

# Calculate the mode of the 5x5 neighbourhood on the valid pixel array
# Use it to smooth the valid pixel array

# Generate the images using the valid pixel array

# SCL
##  0: SC_NODATA
##  1: SC_SATURATED_DEFECTIVE
##  2: SC_DARK_FEATURE_SHADOW
##  3: SC_CLOUD_SHADOW
##  4: SC_VEGETATION
##  5: SC_NOT_VEGETATED
##  6: SC_WATER
##  7: SC_UNCLASSIFIED
##  8: SC_CLOUD_MEDIUM_PROBA
##  9: SC_CLOUD_HIGH_PROBA
## 10: SC_THIN_CIRRUS
## 11: SC_SNOW_ICE


def get_band_paths(safe_folder):
    bands = {
        "10m": {
          "B02": None,
          "B03": None,
          "B04": None,
          "B08": None,
        },
        "20m": {
          "B02": None,
          "B03": None,
          "B04": None,
          "B05": None,
          "B06": None,
          "B07": None,
          "B8A": None,
          "B11": None,
          "B12": None,
          "SCL": None,
        },
        "60m": {
          "B01": None,
          "B02": None,
          "B03": None,
          "B04": None,
          "B05": None,
          "B06": None,
          "B07": None,
          "B8A": None,
          "B09": None,
          "B11": None,
          "B12": None,
          "SCL": None,        
        },
        "QI": {
            'CLDPRB': None
        }
    }
    
    assert os.path.isdir(safe_folder), f"Could not find folder: {safe_folder}"
    
    bands['QI']['CLDPRB'] = glob(f"{safe_folder}/GRANULE/*/QI_DATA/MSK_CLDPRB_20m.jp2")[0]
    
    bands_10m = glob(f"{safe_folder}/GRANULE/*/IMG_DATA/R10m/*_???_*.jp2")
    for band in bands_10m:
        basename = os.path.basename(band)
        band_name = basename.split('_')[2]
        if band_name == 'B02':
            bands['10m']['B02'] = band
        if band_name == 'B03':
            bands['10m']['B03'] = band
        if band_name == 'B04':
            bands['10m']['B04'] = band
        if band_name == 'B08':
            bands['10m']['B08'] = band

    bands_20m = glob(f"{safe_folder}/GRANULE/*/IMG_DATA/R20m/*.jp2")
    for band in bands_20m:
        basename = os.path.basename(band)
        band_name = basename.split('_')[2]
        if band_name == 'B02':
            bands['20m']['B02'] = band
        if band_name == 'B03':
            bands['20m']['B03'] = band
        if band_name == 'B04':
            bands['20m']['B04'] = band
        if band_name == 'B05':
            bands['20m']['B05'] = band
        if band_name == 'B06':
            bands['20m']['B06'] = band
        if band_name == 'B07':
            bands['20m']['B07'] = band
        if band_name == 'B8A':
            bands['20m']['B8A'] = band
        if band_name == 'B09':
            bands['20m']['B09'] = band
        if band_name == 'B11':
            bands['20m']['B11'] = band
        if band_name == 'B12':
            bands['20m']['B12'] = band
        if band_name == 'SCL':
            bands['20m']['SCL'] = band

    bands_60m = glob(f"{safe_folder}/GRANULE/*/IMG_DATA/R60m/*_???_*.jp2")
    for band in bands_60m:
        basename = os.path.basename(band)
        band_name = basename.split('_')[2]
        if band_name == 'B01':
            bands['60m']['B01'] = band
        if band_name == 'B02':
            bands['60m']['B02'] = band
        if band_name == 'B03':
            bands['60m']['B03'] = band
        if band_name == 'B04':
            bands['60m']['B04'] = band
        if band_name == 'B05':
            bands['60m']['B05'] = band
        if band_name == 'B06':
            bands['60m']['B06'] = band
        if band_name == 'B07':
            bands['60m']['B07'] = band
        if band_name == 'B8A':
            bands['60m']['B8A'] = band
        if band_name == 'B09':
            bands['60m']['B09'] = band
        if band_name == 'B11':
            bands['60m']['B11'] = band
        if band_name == 'B12':
            bands['60m']['B12'] = band
        if band_name == 'SCL':
            bands['60m']['SCL'] = band
    
    # Did we get all the bands?
    for resolution in bands:
        for name in bands[resolution]:
            assert bands[resolution][name] != None, f'Could not find band: {safe_folder, resolution, name}'
    
    return bands


def get_metadata(safe_folder):
    metadata = {
        "PRODUCT_START_TIME": None,
        "PRODUCT_STOP_TIME": None,
        "PRODUCT_URI": None,
        "PROCESSING_LEVEL": None,
        "PRODUCT_TYPE": None,
        "PROCESSING_BASELINE": None,
        "GENERATION_TIME": None,
        "SPACECRAFT_NAME": None,
        "DATATAKE_SENSING_START": None,
        "SENSING_ORBIT_NUMBER": None,
        "SENSING_ORBIT_DIRECTION": None,
        "EXT_POS_LIST": None,
        "Cloud_Coverage_Assessment": None,
        "NODATA_PIXEL_PERCENTAGE": None,
        "SATURATED_DEFECTIVE_PIXEL_PERCENTAGE": None,
        "DARK_FEATURES_PERCENTAGE": None,
        "CLOUD_SHADOW_PERCENTAGE": None,
        "VEGETATION_PERCENTAGE": None,
        "NOT_VEGETATED_PERCENTAGE": None,
        "WATER_PERCENTAGE": None,
        "UNCLASSIFIED_PERCENTAGE": None,
        "MEDIUM_PROBA_CLOUDS_PERCENTAGE": None,
        "HIGH_PROBA_CLOUDS_PERCENTAGE": None,
        "THIN_CIRRUS_PERCENTAGE": None,
        "SNOW_ICE_PERCENTAGE": None,
        "folder": safe_folder,
    }

    meta_xml = os.path.join(safe_folder, "MTD_MSIL2A.xml")

    assert os.path.isfile(meta_xml), f"{safe_folder} did not contain a valid metadata file."

    # Parse the xml tree and add metadata
    root = ET.parse(meta_xml).getroot()
    for elem in root.iter():
        if elem.tag in metadata:
            try:
                metadata[elem.tag] = float(elem.text)  # Number?
            except:
                try:
                    metadata[elem.tag] = datetime.datetime.strptime(
                        elem.text, "%Y-%m-%dT%H:%M:%S.%f%z"
                    )  # Date?
                except:
                    metadata[elem.tag] = elem.text

    # Did we get all the metadata?
    for name in metadata:
        assert (
            metadata[name] != None
        ), f"Input metatadata file invalid. {metadata[name]}"

    metadata["INVALID_PERCENTAGE"] = (
        metadata["SATURATED_DEFECTIVE_PIXEL_PERCENTAGE"]
        + metadata["CLOUD_SHADOW_PERCENTAGE"]
        + metadata["MEDIUM_PROBA_CLOUDS_PERCENTAGE"]
        + metadata["HIGH_PROBA_CLOUDS_PERCENTAGE"]
        + metadata["THIN_CIRRUS_PERCENTAGE"]
        + metadata["SNOW_ICE_PERCENTAGE"]
        + (metadata["UNCLASSIFIED_PERCENTAGE"] * 0.25)
        + (metadata["DARK_FEATURES_PERCENTAGE"] * 0.25)
    )
    
    metadata["ALL_INVALID_PERCENTAGE"] = (
        metadata["INVALID_PERCENTAGE"] + metadata['NODATA_PIXEL_PERCENTAGE']
    )
    
    metadata["timestamp"] = float(metadata['DATATAKE_SENSING_START'].timestamp())
    
    metadata["stats"] = {
        "vegetation_count": 0,
        "non_vegetation_count": 0,
        "unclassified_count": 0,
        "water_count": 0,
        "dark_count": 0,
        "other_count": 0,
        "B02": {
            "vegetation": None,
            "vegetation_scale": 1,
            "non_vegetation": None,
            "non_vegetation_scale": 1,
            "unclassified": None,
            "unclassified_scale": 1, 
            "water": None,
            "water_scale": 1,
            "dark": None,
            "dark_scale": 1,
            "other": None,
            "other_scale": 1,
            "scale": 1,
        },
        "B03": {
            "vegetation": None,
            "vegetation_scale": 1,
            "non_vegetation": None,
            "non_vegetation_scale": 1,
            "unclassified": None,
            "unclassified_scale": 1, 
            "water": None,
            "water_scale": 1,
            "dark": None,
            "dark_scale": 1,
            "other": None,
            "other_scale": 1,
            "scale": 1,
        },
        "B04": {
            "vegetation": None,
            "vegetation_scale": 1,
            "non_vegetation": None,
            "non_vegetation_scale": 1,
            "unclassified": None,
            "unclassified_scale": 1, 
            "water": None,
            "water_scale": 1,
            "dark": None,
            "dark_scale": 1,
            "other": None,
            "other_scale": 1,
            "scale": 1,
        },
        "B08": {
            "vegetation": None,
            "vegetation_scale": 1,
            "non_vegetation": None,
            "non_vegetation_scale": 1,
            "unclassified": None,
            "unclassified_scale": 1, 
            "water": None,
            "water_scale": 1,
            "dark": None,
            "dark_scale": 1,
            "other": None,
            "other_scale": 1,
            "scale": 1,
        }
    }

    return metadata


def get_time_difference(dict):
    return dict['time_difference']


def prepare_metadata(list_of_SAFE_images):

    metadata = []
        # Verify files
    for index, image in enumerate(list_of_SAFE_images):
        image_name = os.path.basename(image)
        assert (len(image_name.split("_")) == 7), f"Input file has invalid pattern: {image_name}"
        assert (image_name.rsplit(".")[1] == "SAFE"), f"Input is not a .SAFE folder: {image_name}"

        # Check if / or // or \\ at end of string, if not attach /
        if image.endswith("//"):
            list_of_SAFE_images[index] = image[:-2]

        if image.endswith("\\"):
            list_of_SAFE_images[index] = image[:-2]

        if image.endswith("/"):
            list_of_SAFE_images[index] = image[:-1]

        # Check if safe folder exists
        assert os.path.isdir(list_of_SAFE_images[index]), f"Could not find input folder: {list_of_SAFE_images[index]}"

        # Check if all images are of the same tile.
        if index == 0:
            tile_name = image_name.split("_")[5]
        else:
            this_tile = image_name.split("_")[5]
            assert (tile_name == this_tile), f"Multiple tiles in inputlist: {tile_name}, {this_tile}"

        image_metadata = get_metadata(list_of_SAFE_images[index])
        image_metadata['path'] = get_band_paths(list_of_SAFE_images[index])
        metadata.append(image_metadata)

    lowest_invalid_percentage = 100
    best_image = None
    
    best_images = []
    
    # Find the image with the lowest invalid percentage
    for meta in metadata:
        if meta['ALL_INVALID_PERCENTAGE'] < lowest_invalid_percentage:
            lowest_invalid_percentage = meta['ALL_INVALID_PERCENTAGE']
            best_image = meta

    for meta in metadata:
        if (meta['ALL_INVALID_PERCENTAGE'] - lowest_invalid_percentage) < 3:
            best_images.append(meta)
    
    if len(best_images) != 1:
        # Search the top 3% images for for the most recent
        lowest_mean_b1 = 65535
        for image in best_images:
            nodata_mask = raster_to_array(image['path']['60m']['SCL']) == 0
            b1_mean = np.ma.array(raster_to_array(image['path']['60m']['B01']), mask=nodata_mask).mean()
            if b1_mean < lowest_mean_b1:
                lowest_mean_b1 = b1_mean
                best_image = image

    # Calculate the time difference from each image to the best image
    for meta in metadata:
        meta['time_difference'] = abs(meta['timestamp'] - best_image['timestamp'])

    # Sort by distance to best_image
    metadata = sorted(metadata, key=lambda k: k['time_difference']) 
    
    return metadata

# TODO: create harmonisation function
# TODO: handle all bands
# TODO: add pansharpen
# TODO: ai resample of SWIR

def mosaic_tile(
    list_of_SAFE_images,
    out_dir,
    out_name='mosaic',
    dst_projection=None,
    feather=True,
    cutoff_invalid=1,
    cutoff_cloud=2,
    invalid_contract=11,
    invalid_expand=51,
    border_dist=61,
    feather_dist=21,
    filter_tracking=True,
    filter_tracking_dist=7,
    filter_tracking_iterations=2,
    match_mean=True,
    match_quintile=0.25,
    match_individual_scaling=False,
    max_days=30,
    max_images_include=10,
    max_search_images=15,
):

    # Verify input
    assert isinstance(list_of_SAFE_images, list), "list_of_SAFE_images is not a list. [path_to_safe_file1, path_to_safe_file2, ...]"
    assert isinstance(out_dir, str), f"out_dir is not a string: {out_dir}"
    assert isinstance(out_name, str), f"out_name is not a string: {out_name}"
    assert len(list_of_SAFE_images) > 1, "list_of_SAFE_images is empty or only a single image."

    print('Selecting best image..')
    metadata = prepare_metadata(list_of_SAFE_images)
    
    # Sorted by best, so 0 is the best one.
    best_image = metadata[0]
    best_image_folder = best_image['folder']
    print(f'Selected: {os.path.basename(os.path.normpath(best_image_folder))}')

    time_limit = (max_days * 86400)
    
    print('Resampling and reading base image..')
    border_kernel = create_kernel(border_dist, weighted_edges=False, weighted_distance=False, normalise=False).astype('uint8')
    invalid_kernel = create_kernel(invalid_expand, weighted_edges=False, weighted_distance=False, normalise=False).astype('uint8')
    invalid_kernel_small = create_kernel(invalid_contract, weighted_edges=False, weighted_distance=False, normalise=False).astype('uint8')
    
    scls = {}
    
    # Resample the tracking landcover array
    slc = raster_to_array(resample(best_image['path']['20m']['SCL'], reference_raster=best_image['path']['10m']['B04'])).astype('uint8')
    scls["0"] = slc
    nodata = cv2.dilate(np.where(slc == 0, 1, 0).astype('uint8'), border_kernel, iterations=1).astype('bool')
    cldprb = (raster_to_array(resample(best_image['path']['QI']['CLDPRB'], reference_raster=best_image['path']['10m']['B04'])) > cutoff_cloud).astype('bool')

    mask = ((slc == 1) | (slc == 3))
    mask = np.logical_or(mask, nodata)
    mask = np.logical_or(mask, cldprb)
    mask = cv2.erode(mask.astype('uint8'), invalid_kernel_small)
    mask = cv2.dilate(mask.astype('uint8'), invalid_kernel).astype('bool')
    
    # Match the means of the different tiles per classification type.
    if match_mean:
        vegetation_means = { "B02": [], "B03": [], "B04": [], "B08": [] }
        non_vegetation_means = { "B02": [], "B03": [], "B04": [], "B08": [] }
        water_means = { "B02": [], "B03": [], "B04": [], "B08": [] }
        unclassified_means = { "B02": [], "B03": [], "B04": [], "B08": [] }
        dark_means = { "B02": [], "B03": [], "B04": [], "B08": [] }
        other_means = { "B02": [], "B03": [], "B04": [], "B08": [] }

        vegetation_mask = np.logical_or(slc != 4, mask)
        non_vegetation_mask = np.logical_or(slc != 5, mask)
        water_mask = np.logical_or(slc != 6, mask)
        unclassified_mask = np.logical_or(slc != 7, mask)
        dark_mask = np.logical_or((slc != 2) & (slc != 3), mask)
        other_mask = np.logical_or((slc != 1) & (slc != 8) & (slc != 9) & (slc != 10) & (slc != 11), mask)

        metadata[0]['stats']['vegetation_count'] = (vegetation_mask == False).sum()
        metadata[0]['stats']['non_vegetation_count'] = (non_vegetation_mask == False).sum()
        metadata[0]['stats']['water_count'] = (water_mask == False).sum()
        metadata[0]['stats']['unclassified_count'] = (unclassified_mask == False).sum()
        metadata[0]['stats']['dark_count'] = (dark_mask == False).sum()
        metadata[0]['stats']['other_count'] = (other_mask == False).sum()

        for band in ['B02', 'B03', 'B04', 'B08']:
            base_ref = raster_to_array(metadata[0]['path']['10m'][band])
            
            if metadata[0]['stats']['vegetation_count'] == 0:
                metadata[0]['stats'][band]['vegetation'] = -1
            else:
                metadata[0]['stats'][band]['vegetation'] = np.ma.array(base_ref, mask=vegetation_mask).mean()
                vegetation_means[band].append(metadata[0]['stats'][band]['vegetation'])
            
            if metadata[0]['stats']['non_vegetation_count'] == 0:
                metadata[0]['stats'][band]['non_vegetation'] = -1
            else:
                metadata[0]['stats'][band]['non_vegetation'] = np.ma.array(base_ref, mask=non_vegetation_mask).mean()
                non_vegetation_means[band].append(metadata[0]['stats'][band]['non_vegetation'])
            
            if metadata[0]['stats']['water_count'] == 0:
                metadata[0]['stats'][band]['water'] = -1
            else:    
                metadata[0]['stats'][band]['water'] = np.ma.array(base_ref, mask=water_mask).mean()
                water_means[band].append(metadata[0]['stats'][band]['water'])
            
            if metadata[0]['stats']['unclassified_count'] == 0:
                metadata[0]['stats'][band]['unclassified'] = -1
            else: 
                metadata[0]['stats'][band]['unclassified'] = np.ma.array(base_ref, mask=unclassified_mask).mean()
                unclassified_means[band].append(metadata[0]['stats'][band]['unclassified'])
        
            if  metadata[0]['stats']['dark_count'] == 0:
                metadata[0]['stats'][band]['dark'] = -1
            else: 
                metadata[0]['stats'][band]['dark'] = np.ma.array(base_ref, mask=dark_mask).mean()
                dark_means[band].append(metadata[0]['stats'][band]['dark'])
        
            if metadata[0]['stats']['other_count'] == 0:
                metadata[0]['stats'][band]['other'] = -1
            else: 
                metadata[0]['stats'][band]['other'] = np.ma.array(base_ref, mask=other_mask).mean()
                other_means[band].append(metadata[0]['stats'][band]['other'])

        vegetation_mask = None
        non_vegetation_mask = None
        water_mask = None
        unclassified_mask = None
        dark_mask = None
        other_mask = None

    # Clear memory
    nodata = None
    cldprb = None
    
    metadata[0]['valid_mask'] = mask

    tracking_array = np.zeros(mask.shape, dtype='uint8')
      
    coverage = (mask.sum() / mask.size) * 100
    current_image_index = 1  # The 0 index is for the best image
    
    processed_images_indices = [0]
       
    print(f'Initial. tracking array: {round(coverage, 3)} towards goal: {cutoff_invalid}')
    # Loop the images and update the tracking array
    while coverage > cutoff_invalid and current_image_index < len(metadata) - 1 and len(processed_images_indices) <= max_images_include and current_image_index <= max_search_images and metadata[current_image_index]['time_difference'] < time_limit:
        current_metadata = metadata[current_image_index]
        ex_slc = raster_to_array(resample(current_metadata['path']['20m']['SCL'], reference_raster=current_metadata['path']['10m']['B04'])).astype('uint8')
        scls[str(current_image_index)] = ex_slc
        ex_nodata = cv2.dilate(np.where(ex_slc == 0, 1, 0).astype('uint8'), border_kernel, iterations=1).astype('bool')
        ex_cldprb = (raster_to_array(resample(current_metadata['path']['QI']['CLDPRB'], reference_raster=current_metadata['path']['10m']['B04'])) > cutoff_cloud).astype('bool')
        
        ex_mask = ((ex_slc == 1) | (ex_slc == 3))
        ex_mask = np.logical_or(ex_mask, ex_nodata)
        ex_mask = np.logical_or(ex_mask, ex_cldprb)
        ex_mask = cv2.erode(ex_mask.astype('uint8'), invalid_kernel_small).astype('bool')
        ex_mask = cv2.dilate(ex_mask.astype('uint8'), invalid_kernel).astype('bool')
        
        change_mask = (np.logical_and(mask == True, np.logical_and((mask != ex_mask), (ex_mask == False)))).astype('bool')
        
        # Only process if change is more that 0.5 procent.
        if ((change_mask.sum() / change_mask.size) * 100) > 0.5:
            if match_mean:
                ex_vegetation_mask = np.logical_or(ex_slc != 4, change_mask == False)
                ex_non_vegetation_mask = np.logical_or(ex_slc != 5, change_mask == False)
                ex_water_mask = np.logical_or(ex_slc != 6, change_mask == False)
                ex_unclassified_mask = np.logical_or(ex_slc != 7, change_mask == False)
                ex_dark_mask = np.logical_or((ex_slc != 2) & (ex_slc != 3), change_mask == False)
                ex_other_mask = np.logical_or((ex_slc != 1) & (ex_slc != 8) & (ex_slc != 9) & (ex_slc != 10) & (ex_slc != 11), change_mask == False)

                metadata[current_image_index]['stats']['vegetation_count'] = (ex_vegetation_mask == False).sum()
                metadata[current_image_index]['stats']['non_vegetation_count'] = (ex_non_vegetation_mask == False).sum()
                metadata[current_image_index]['stats']['water_count'] = (ex_water_mask == False).sum()
                metadata[current_image_index]['stats']['unclassified_count'] = (ex_unclassified_mask == False).sum()
                metadata[current_image_index]['stats']['dark_count'] = (ex_dark_mask == False).sum()
                metadata[current_image_index]['stats']['other_count'] = (ex_other_mask == False).sum()
                
                for band in ['B02', 'B03', 'B04', 'B08']:              
                    ex_ref = raster_to_array(metadata[current_image_index]['path']['10m'][band])
                    
                    if metadata[current_image_index]['stats']['vegetation_count'] == 0:
                        metadata[current_image_index]['stats'][band]['vegetation'] = -1
                    else:
                        metadata[current_image_index]['stats'][band]['vegetation'] = np.ma.array(ex_ref, mask=ex_vegetation_mask).mean()
                        vegetation_means[band].append(metadata[current_image_index]['stats'][band]['vegetation'])
                    
                    if metadata[current_image_index]['stats']['non_vegetation_count'] == 0:
                        metadata[current_image_index]['stats'][band]['non_vegetation'] = -1
                    else:
                        metadata[current_image_index]['stats'][band]['non_vegetation'] = np.ma.array(ex_ref, mask=ex_non_vegetation_mask).mean()
                        non_vegetation_means[band].append(metadata[current_image_index]['stats'][band]['non_vegetation'])

                    if metadata[current_image_index]['stats']['water_count'] == 0:
                        metadata[current_image_index]['stats'][band]['water'] = -1
                    else:
                        metadata[current_image_index]['stats'][band]['water'] = np.ma.array(ex_ref, mask=ex_water_mask).mean()
                        water_means[band].append(metadata[current_image_index]['stats'][band]['water'])
                    
                    if metadata[current_image_index]['stats']['unclassified_count'] == 0:
                        metadata[current_image_index]['stats'][band]['unclassified'] = -1
                    else:
                        metadata[current_image_index]['stats'][band]['unclassified'] = np.ma.array(ex_ref, mask=ex_unclassified_mask).mean()
                        unclassified_means[band].append(metadata[current_image_index]['stats'][band]['unclassified'])
                    
                    if metadata[current_image_index]['stats']['dark_count'] == 0:
                        metadata[current_image_index]['stats'][band]['dark'] = -1
                    else: 
                        metadata[current_image_index]['stats'][band]['dark'] = np.ma.array(ex_ref, mask=ex_dark_mask).mean()
                        dark_means[band].append(metadata[current_image_index]['stats'][band]['dark'])
                
                    if metadata[current_image_index]['stats']['other_count'] == 0:
                        metadata[current_image_index]['stats'][band]['other'] = -1
                    else: 
                        metadata[current_image_index]['stats'][band]['other'] = np.ma.array(ex_ref, mask=ex_other_mask).mean()
                        other_means[band].append(metadata[current_image_index]['stats'][band]['other'])

                ex_vegetation_mask = None
                ex_non_vegetation_mask = None
                ex_water_mask = None
                ex_unclassified_mask = None
                ex_dark_mask = None
                ex_other_mask = None

            # Clear memory
            ex_nodata = None
            ex_cldprb = None

            metadata[current_image_index]['valid_mask'] = ex_mask
            mask = np.logical_and(mask, ex_mask).astype('bool')
            
            # Add to tracking array and slc
            tracking_array = tracking_array + (change_mask * current_image_index)
            slc = np.where(change_mask, ex_slc, slc)

            # Update coverage
            coverage = (mask.sum() / mask.size) * 100

            processed_images_indices.append(current_image_index)

            print(f'Updating tracking array: {round(coverage, 3)} towards goal: {cutoff_invalid}')
        else:
            print('Skipping image due to low change.. (0.5% threshold)')

        current_image_index += 1      

    # Only merge images if there are more than one.
    if len(processed_images_indices) > 1:

        if match_mean is True:
            # Calculate the scaling factors
            sums = {
                "vegetation": 0,
                "non_vegetation": 0,
                "water": 0,
                "unclassified": 0,
                "dark": 0,
                "other": 0,
            }
                      
            for i in processed_images_indices:
                img = metadata[i]
                sums['vegetation'] += img['stats']['vegetation_count']
                sums['non_vegetation'] += img['stats']['non_vegetation_count']
                sums['water'] += img['stats']['water_count']
                sums['unclassified'] += img['stats']['unclassified_count']
                sums['dark'] += img['stats']['dark_count']
                sums['other'] += img['stats']['other_count']
            
            total_pixels = 0
            for key in sums:
                if key != 'other':
                    total_pixels += sums[key]
            
            weights = {
                'vegetation': [],
                'non_vegetation': [],
                'water': [],
                'unclassified': [],
                'dark': [],
                'other': [],
            }
            
            for i in processed_images_indices:
                img = metadata[i]
                weights["vegetation"].append(img['stats']['vegetation_count'] / sums['vegetation'])
                weights["non_vegetation"].append(img['stats']['non_vegetation_count'] / sums['non_vegetation'])
                weights["water"].append(img['stats']['water_count'] / sums['water'])
                weights["unclassified"].append(img['stats']['unclassified_count'] / sums['unclassified'])
                weights["dark"].append(img['stats']['dark_count'] / sums['dark'])
                weights["other"].append(img['stats']['other_count'] / sums['other'])


            quintiles = { "B02": {}, "B03": {}, "B04": {}, "B08": {} }

            for band in ['B02', 'B03', 'B04', 'B08']:
                quintiles[band]['vegetation'] = weighted_quantile(vegetation_means[band], match_quintile, weights['vegetation'])
                quintiles[band]['non_vegetation'] = weighted_quantile(non_vegetation_means[band], match_quintile, weights['non_vegetation'])
                quintiles[band]['water'] = weighted_quantile(water_means[band], match_quintile, weights['water'])
                quintiles[band]['unclassified'] = weighted_quantile(unclassified_means[band], match_quintile, weights['unclassified'])
                quintiles[band]['dark'] = weighted_quantile(dark_means[band], match_quintile, weights['dark'])
                quintiles[band]['other'] = weighted_quantile(other_means[band], match_quintile, weights['other'])

            for i in processed_images_indices:
                for band in ['B02', 'B03', 'B04', 'B08']:

                    # If the is no pixels at all of the class. Insert a zero. Value does not matter as addition will be zero.
                    if len(vegetation_means[band]) == 0: vegetation_means[band].append(0)
                    if len(non_vegetation_means[band]) == 0: non_vegetation_means[band].append(0)
                    if len(water_means[band]) == 0: water_means[band].append(0)
                    if len(unclassified_means[band]) == 0: unclassified_means[band].append(0)
                    if len(dark_means[band]) == 0: dark_means[band].append(0)
                    if len(other_means[band]) == 0: other_means[band].append(0)

                    # If there is no data of the classification, use the median value of all other.
                    if metadata[i]['stats'][band]['vegetation'] == -1:
                        metadata[i]['stats'][band]['vegetation'] = weighted_quantile(vegetation_means[band], 0.5, weights['vegetation'])
                    if metadata[i]['stats'][band]['non_vegetation'] == -1:
                        metadata[i]['stats'][band]['non_vegetation'] = weighted_quantile(non_vegetation_means[band], 0.5, weights['non_vegetation'])
                    if metadata[i]['stats'][band]['water'] == -1:
                        metadata[i]['stats'][band]['water'] = weighted_quantile(water_means[band], 0.5, weights['water'])
                    if metadata[i]['stats'][band]['unclassified'] == -1:
                        metadata[i]['stats'][band]['unclassified'] = weighted_quantile(unclassified_means[band], 0.5, weights['unclassified'])
                    if metadata[i]['stats'][band]['dark'] == -1:
                        metadata[i]['stats'][band]['dark'] = weighted_quantile(dark_means[band], 0.5, weights['dark'])
                    if metadata[i]['stats'][band]['other'] == -1:
                        metadata[i]['stats'][band]['other'] = weighted_quantile(other_means[band], 0.5, weights['other'])

                    metadata[i]['stats'][band]['vegetation_scale'] = quintiles[band]['vegetation'] / metadata[i]['stats'][band]['vegetation']
                    metadata[i]['stats'][band]['non_vegetation_scale'] = quintiles[band]['non_vegetation'] / metadata[i]['stats'][band]['non_vegetation']
                    metadata[i]['stats'][band]['water_scale'] = quintiles[band]['water'] / metadata[i]['stats'][band]['water']
                    metadata[i]['stats'][band]['unclassified_scale'] = quintiles[band]['unclassified'] / metadata[i]['stats'][band]['unclassified']
                    metadata[i]['stats'][band]['dark_scale'] = quintiles[band]['dark'] / metadata[i]['stats'][band]['dark']
                    metadata[i]['stats'][band]['other_scale'] = quintiles[band]['other'] / metadata[i]['stats'][band]['other']

            if match_individual_scaling is False:

                simple_weights = {
                    'vegetation': sums['vegetation'] / total_pixels,
                    'non_vegetation': sums['non_vegetation'] / total_pixels,
                    'water': sums['water'] / total_pixels,
                    'unclassified': sums['unclassified'] / total_pixels,
                    'dark': sums['dark'] / total_pixels,
                }

                for i in processed_images_indices:
                    for band in ['B02', 'B03', 'B04', 'B08']:
                        simple_scale = (
                            (simple_weights['vegetation'] * metadata[i]['stats'][band]['vegetation_scale'])
                          + (simple_weights['non_vegetation'] * metadata[i]['stats'][band]['non_vegetation_scale'])
                          + (simple_weights['water'] *  metadata[i]['stats'][band]['water_scale'])
                          + (simple_weights['unclassified'] * metadata[i]['stats'][band]['unclassified_scale'])
                          + (simple_weights['dark'] * metadata[i]['stats'][band]['dark_scale'])
                        )

                        metadata[i]['stats'][band]['scale'] = simple_scale

        if filter_tracking is True:
            print('Filtering tracking array..')
            # Run a mode filter on the tracking array
            tracking_array = mode_filter(tracking_array, filter_tracking_dist, filter_tracking_iterations).astype('uint8')
        
        array_to_raster(tracking_array.astype('uint8'), reference_raster=best_image['path']['10m']['B08'], out_raster=os.path.join(out_dir, f"tracking_{out_name}.tif"), dst_projection=dst_projection)

        if feather:
            print('Precalculating feathers..')
            feathers = {}
            for i in processed_images_indices:
                feathers[str(i)] = feather_s2_filter(tracking_array, i, feather_dist).astype('float32')


    # BUG: Output SLC is wrong.
    # Save the slc file.
    array_to_raster(slc.astype('uint8'), reference_raster=best_image['path']['10m']['B08'], out_raster=os.path.join(out_dir, f"slc_{out_name}.tif"), dst_projection=dst_projection)


    # BUG: Scaling is broken.
    # TODO: Add nodata to output.
    bands_to_output = ['B02']
    # bands_to_output = ['B02', 'B03', 'B04', 'B08']

    print('Merging band data..')
    if match_mean is True and match_individual_scaling is True:

        for band in bands_to_output:

            print(f'Writing: {band}..')

            base_image = raster_to_array(metadata[0]['path']['10m'][band]).astype('float32')

            if len(processed_images_indices) > 1:

                for i in processed_images_indices:
                    if i == 0:
                        base_image = np.where((tracking_array == 0) & (scls[str(0)] == 4), base_image * metadata[0]['stats'][band]['vegetation_scale'], base_image)
                        base_image = np.where((tracking_array == 0) & (scls[str(0)] == 5), base_image * metadata[0]['stats'][band]['non_vegetation_scale'], base_image)
                        base_image = np.where((tracking_array == 0) & (scls[str(0)] == 6), base_image * metadata[0]['stats'][band]['water_scale'], base_image)
                        base_image = np.where((tracking_array == 0) & (scls[str(0)] == 7), base_image * metadata[0]['stats'][band]['unclassified_scale'], base_image)
                        base_image = np.where((tracking_array == 0) & ((scls[str(0)] != 2) & (scls[str(0)] != 3)), base_image * metadata[i]['stats'][band]['dark_scale'], base_image)
                        base_image = np.where((tracking_array == 0) & ((scls[str(0)] != 1) & (scls[str(0)] != 8) & (scls[str(0)] != 9) & (scls[str(0)] != 10) & (scls[str(0)] != 11)), base_image * metadata[0]['stats'][band]['other_scale'], base_image)
                        
                        if feather is True:
                            base_image = base_image * feathers[str(i)]
                    else:
                        add_band = raster_to_array(metadata[i]['path']['10m'][band]).astype('float32')
                        add_band = np.where((tracking_array == i) & (scls[str(i)] == 4), add_band * metadata[i]['stats'][band]['vegetation_scale'], add_band)
                        add_band = np.where((tracking_array == i) & (scls[str(i)] == 5), add_band * metadata[i]['stats'][band]['non_vegetation_scale'], add_band)
                        add_band = np.where((tracking_array == i) & (scls[str(i)] == 6), add_band * metadata[i]['stats'][band]['water_scale'], add_band)
                        add_band = np.where((tracking_array == i) & (scls[str(i)] == 7), add_band * metadata[i]['stats'][band]['unclassified_scale'], add_band)
                        add_band = np.where((tracking_array == i) & ((scls[str(i)] != 2) & (scls[str(i)] != 3)), add_band * metadata[i]['stats'][band]['dark_scale'], add_band)
                        add_band = np.where((tracking_array == i) & ((scls[str(i)] != 1) & (scls[str(i)] != 8) & (scls[str(i)] != 9) & (scls[str(i)] != 10) & (scls[str(i)] != 11)), add_band * metadata[i]['stats'][band]['other_scale'], add_band)

                        if feather is True:
                            base_image = np.add(base_image, (add_band * feathers[str(i)]))
                        else:
                            base_image = np.where(tracking_array == i, add_band, base_image).astype('float32')
                
            array_to_raster(base_image.astype('uint16'), reference_raster=best_image['path']['10m'][band], out_raster=os.path.join(out_dir, f"{band}_{out_name}.tif"), dst_projection=dst_projection)
    else:
        for band in bands_to_output:
            print(f'Writing: {band}..')
            base_image = raster_to_array(best_image['path']['10m'][band]).astype('float32')

            for i in processed_images_indices:
                if i == 0:
                    if match_mean and len(processed_images_indices) > 1:
                        base_image = base_image * best_image['stats'][band]['scale']

                    if feather is True and len(processed_images_indices) > 1:
                        base_image = base_image * feathers[str(i)]

                else:
                    add_band = raster_to_array(metadata[i]['path']['10m'][band]).astype('float32')
                    
                    if match_mean:
                        add_band = add_band * metadata[i]['stats'][band]['scale']
                
                    if feather is True:
                        base_image = np.add(base_image, (add_band * feathers[str(i)]))
                    else:
                        base_image = np.where(tracking_array == i, add_band, base_image).astype('float32')
            
            array_to_raster(base_image.astype('uint16'), reference_raster=best_image['path']['10m'][band], out_raster=os.path.join(out_dir, f"{band}_{out_name}.tif"), dst_projection=dst_projection)


if __name__ == "__main__":
    folder = "/mnt/c/Users/caspe/Desktop/tmp/"
    out_dir = "/mnt/c/Users/caspe/Desktop/tmp/mosaic/"
    images = glob(folder + "*.*")
    
    mosaic_tile(
        images,
        out_dir,
        "mosaic",
    )

