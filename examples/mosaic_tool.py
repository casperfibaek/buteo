import sys; sys.path.append('..'); sys.path.append('../lib/')
from lib.raster_io import raster_to_array, array_to_raster
from lib.raster_resample import resample
from lib.stats_filters import mode_filter, feather_s2_filter
from lib.stats_kernel import create_kernel
from lib.utils_core import weighted_quantile
from time import time
import cv2
import os
import xml.etree.ElementTree as ET
import datetime
from glob import glob
import numpy as np

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
    
    
    # import pdb; pdb.set_trace()

    # Did we get all the bands?
    # for resolution in bands:
    #     for name in bands[resolution]:
    #         assert bands[resolution][name] != None, f'Could not find band: {safe_folder, resolution, name}'
    
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
        + (metadata["DARK_FEATURES_PERCENTAGE"] * 0.75)
        + (metadata["UNCLASSIFIED_PERCENTAGE"] * 0.25)
    )
    
    metadata["VALID_LAND"] = (
        metadata["VEGETATION_PERCENTAGE"]
        + metadata["NOT_VEGETATED_PERCENTAGE"]
        + (metadata["UNCLASSIFIED_PERCENTAGE"] * 0.75)
        + (metadata["DARK_FEATURES_PERCENTAGE"] * 0.25)
    )
    
    metadata["INVALID_LAND"] = 100 - metadata["VALID_LAND"]
    
    metadata["ALL_INVALID_PERCENTAGE"] = (
        metadata["INVALID_PERCENTAGE"] + metadata['NODATA_PIXEL_PERCENTAGE']
    )
    
    metadata["timestamp"] = float(metadata['DATATAKE_SENSING_START'].timestamp())

    return metadata

def get_time_difference(dict):
    return dict['time_difference']

def assess_radiometric_quality(metadata, quality='high'):
    if quality == 'high':
        scl = raster_to_array(resample(metadata['path']['20m']['SCL'], reference_raster=metadata['path']['10m']['B04'])).astype('uint8')
        band_01 = raster_to_array(resample(metadata['path']['60m']['B01'], reference_raster=metadata['path']['10m']['B04'])).astype('uint16')
        band_02 = raster_to_array(metadata['path']['10m']['B02']).astype('uint16')
    else:
        scl = raster_to_array(metadata['path']['60m']['SCL']).astype('uint8')
        band_01 = raster_to_array(metadata['path']['60m']['B01']).astype('uint16')
        band_02 = raster_to_array(metadata['path']['60m']['B02']).astype('uint16')
   
    # Scale goes 0 best, 100 worst
    quality = np.zeros(band_02.shape).astype('uint8')
    
    # Dilate nodata values by 1km each side 
    kernel = create_kernel(201, weighted_edges=False, weighted_distance=False, normalise=False).astype('uint8')
    nodata_dilated = cv2.dilate((scl == 0).astype('uint8'), kernel).astype('uint8')
    
    quality = np.where((nodata_dilated == 1), 100, quality)
    quality = np.where((scl == 1), 100, quality)
    quality = np.where(np.logical_or((scl == 2), (scl == 3)), 90, quality)
    quality = np.where((scl == 9), 90, quality)
    quality = np.where((scl == 8), 70, quality)
    quality = np.where((scl == 11), 50, quality)
    quality = np.where((scl == 10), 2, quality)

    quality = np.where((band_01 > 885), quality + 10, quality)
    quality = np.where(np.logical_and((band_01 > 600), (band_01 <= 850)), quality + 2, quality)
    quality = np.where(np.logical_and((scl != 0), np.logical_and((band_01 <= 1), (band_02 <= 1))), quality + 15, quality)
    quality = np.where(np.logical_and((scl != 0), np.logical_and((band_01 <= 10), (band_02 <= 200))), quality + 5, quality)

    return quality, scl, band_01

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
        if meta['INVALID_LAND'] < lowest_invalid_percentage:
            lowest_invalid_percentage = meta['INVALID_LAND']
            best_image = meta

    for meta in metadata:
        if (meta['INVALID_LAND'] - lowest_invalid_percentage) < 5:
            best_images.append(meta)

    if len(best_images) != 1:
        # Search the top 5% images for the best image
        min_quality_score = 1000000000000000000
        for image in best_images:
            quality, scl, b1 = assess_radiometric_quality(image, quality='low')

            quality_score = np.ma.masked_where((scl == 4) | (scl == 5) | (scl == 7), 1000 - quality, 1000).sum()

            if quality_score < min_quality_score:
                min_quality_score = quality_score
                best_image = image


    # Calculate the time difference from each image to the best image
    for meta in metadata:
        meta['time_difference'] = abs(meta['timestamp'] - best_image['timestamp'])

    # Sort by distance to best_image
    metadata = sorted(metadata, key=lambda k: k['time_difference']) 
    
    return metadata

# TODO: Find out what is wrong with: ['30NYL', '30PWR', '30PXR', '30PXS', '30PYQ', '30NWN', '30NZM', '30NZP']
# TODO: Move harmonisation function to 60m
# TODO: Add multiprocessing
# TODO: Add overlap harmonisation
# TODO: handle all bands
# TODO: add pansharpen
# TODO: ai resample of SWIR

def mosaic_tile(
    list_of_SAFE_images,
    out_dir,
    out_name='mosaic',
    dst_projection=None,
    feather=True,
    invalid_threshold=1,
    vapour_ratio=0.75,
    vapour_tests=True,
    feather_dist=15,
    filter_tracking=True,
    filter_tracking_dist=9,
    filter_tracking_iterations=1,
    match_mean=True,
    match_quintile=0.25,
    max_days=30,
    max_images_include=15,
    max_search_images=35,
):

    start_time = time()

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

    
    print('Resampling and reading base image..')
    quality, scl, b1 = assess_radiometric_quality(best_image)
    
    time_limit = (max_days * 86400)
    pixel_count = quality.size
    
    kernel_contract = create_kernel(5, weighted_edges=False, weighted_distance=False, normalise=False).astype('uint8')
    kernel_contract_dilate = create_kernel(3, weighted_edges=False, weighted_distance=False, normalise=False).astype('uint8')
    kernel_dilate = create_kernel(15, weighted_edges=False, weighted_distance=False, normalise=False).astype('uint8')
    
    mask = (quality > 0).astype('uint8')
    mask = cv2.dilate(mask, kernel_dilate).astype('bool')
       
    tracking_array = np.zeros(mask.shape, dtype='uint8')
    timing_array = np.zeros(mask.shape, dtype='uint16')
      
    coverage = ((quality == 0).sum() / quality.size) * 100
    current_image_index = 1  # The 0 index is for the best image
    processed_images_indices = [0]

    # Match the means of the different tiles per classification type.

    print(f'Initial. tracking array: {round(coverage, 2)} towards goal: {invalid_threshold}')
    # Loop the images and update the tracking array
    while (100 - coverage) > invalid_threshold and current_image_index < len(metadata) - 1 and len(processed_images_indices) <= max_images_include and current_image_index <= max_search_images and (metadata[current_image_index]['time_difference'] < time_limit or (scl == 0).sum() > 0):
        current_metadata = metadata[current_image_index]
        ex_quality, ex_scl, ex_b1 = assess_radiometric_quality(current_metadata)
        
        # Time difference
        td = int(round(metadata[current_image_index]['time_difference'] / 86400, 0))
        time_difference = (np.full(timing_array.shape, td) - timing_array).astype('uint16')
        
        ex_mask = (ex_quality >= quality).astype('uint8')
        ex_mask = cv2.dilate(ex_mask, kernel_dilate).astype('bool')

        if vapour_tests is True:
            # Don't add pixels that are significantly brighter, add pixels that are slightly darker
            with np.errstate(invalid='ignore'):
                with np.errstate(divide='ignore'):
                    b1_ratio = (ex_b1 / b1)
                    equal_quality = (ex_quality <= quality)
                    b1_mask_add = ((b1_ratio < vapour_ratio) & (equal_quality & (time_difference <= 7))).astype('uint8')
                    b1_mask_add = cv2.erode(b1_mask_add, kernel_contract)
                    b1_mask_add = cv2.dilate(b1_mask_add, kernel_contract_dilate).astype('bool')
                    b1_mask_remove = (b1_ratio <= (1 + (1 - vapour_ratio)))

            change_mask = mask & b1_mask_remove & ((ex_mask == False) | b1_mask_add)
            # change_mask = mask & b1_mask_remove & (ex_mask == False)
        else:
            change_mask = mask & (ex_mask == False)
        
        # Only process if change is more than 0.5 procent.
        if ((change_mask.sum() / change_mask.size) * 100) > 0.5:
            
            # Udpdate the trackers
            timing_array = np.where(change_mask, td, timing_array).astype('uint16')
            tracking_array = np.where(change_mask, current_image_index, tracking_array).astype('uint8')
            scl = np.where(change_mask, ex_scl, scl).astype('uint8')
            quality = np.where(change_mask, ex_quality, quality).astype('uint8')
            
            if vapour_tests is True:
                b1 = np.where(change_mask, ex_b1, b1).astype('uint16')

            mask = (quality > 0).astype('uint8')

            # Update coverage
            coverage = ((quality == 0).sum() / quality.size) * 100

            processed_images_indices.append(current_image_index)

            img_name = os.path.basename(os.path.normpath(metadata[current_image_index]['folder'])).split('_')[-1].split('.')[0]
            print(f'Updating tracking array: {round(coverage, 2)}, goal: {100 - invalid_threshold}, img: {img_name}, {td}/{max_days} days')
        else:
            print(f'Skipping image due to low change.. (0.5% threshold) ({td}/{max_days} days)')

        current_image_index += 1

    # Only merge images if there are more than one.
    if len(processed_images_indices) > 1:

        if match_mean is True:
            print('Harmonising layers..')
        
            for i in processed_images_indices:                
                metadata[i]['stats'] = {
                    'B02': { 'vegetation': 0, 'not_vegetation': 0, 'unclassified': 0, 'scale': 1 },
                    'B03': { 'vegetation': 0, 'not_vegetation': 0, 'unclassified': 0, 'scale': 1 },
                    'B04': { 'vegetation': 0, 'not_vegetation': 0, 'unclassified': 0, 'scale': 1 },
                    'B08': { 'vegetation': 0, 'not_vegetation': 0, 'unclassified': 0, 'scale': 1 },
                    'counts': {
                        'vegetation': 0,
                        'not_vegetation': 0,
                        'unclassified': 0
                    },
                }
                
                # Prepare masks
                layer_mask = tracking_array == i
                layer_veg = layer_mask & (scl == 4)
                layer_non_veg = layer_mask & (scl == 5)
                layer_unclassified = layer_mask & (scl == 7)
                
                # Calculate count of pixels
                metadata[i]['stats']['counts']['vegetation'] = layer_veg.sum()
                metadata[i]['stats']['counts']['not_vegetation'] = layer_non_veg.sum()
                metadata[i]['stats']['counts']['unclassified'] = layer_unclassified.sum()

                # Calculate the mean values for each band
                for band in ['B02', 'B03', 'B04', 'B08']:
                    array = raster_to_array(metadata[i]['path']['10m'][band])
                    metadata[i]['stats'][band]['vegetation'] = np.ma.array(array, mask=layer_veg == False).mean()
                    metadata[i]['stats'][band]['not_vegetation'] = np.ma.array(array, mask=layer_non_veg == False).mean()
                    metadata[i]['stats'][band]['unclassified'] = np.ma.array(array, mask=layer_unclassified == False).mean()

            pixel_sums = {
                'vegetation': 0,
                'not_vegetation': 0,
                'unclassified': 0,
                'valid_ground_pixels': 0,             
            }
            
            mean_values = {
                'B02': { 'vegetation': [], 'not_vegetation': [], 'unclassified': [] },
                'B03': { 'vegetation': [], 'not_vegetation': [], 'unclassified': [] },
                'B04': { 'vegetation': [], 'not_vegetation': [], 'unclassified': [] },
                'B08': { 'vegetation': [], 'not_vegetation': [], 'unclassified': [] },
            }
            
            for i in processed_images_indices:
                image = metadata[i]
                counts = [
                    image['stats']['counts']['vegetation'],
                    image['stats']['counts']['not_vegetation'],
                    image['stats']['counts']['unclassified'],
                ]
                pixel_sums['vegetation'] += counts[0]
                pixel_sums['not_vegetation'] += counts[1]
                pixel_sums['unclassified'] += counts[2]
                pixel_sums['valid_ground_pixels'] += sum(counts)
                
                for band in ['B02', 'B03', 'B04', 'B08']:
                    mean_values[band]['vegetation'].append(image['stats'][band]['vegetation'])
                    mean_values[band]['not_vegetation'].append(image['stats'][band]['not_vegetation'])
                    mean_values[band]['unclassified'].append(image['stats'][band]['unclassified'])

            ratios = {
                'vegetation': pixel_sums['vegetation'] / pixel_sums['valid_ground_pixels'],
                'not_vegetation': pixel_sums['not_vegetation'] / pixel_sums['valid_ground_pixels'],
                'unclassified': pixel_sums['unclassified'] / pixel_sums['valid_ground_pixels'],
            }

            weights = { 'vegetation': [], 'not_vegetation': [], 'unclassified': [] }
            
            for i in processed_images_indices:
                image = metadata[i]

                if pixel_sums['vegetation'] == 0:
                    weights['vegetation'].append(0)
                else:
                    weights['vegetation'].append(image['stats']['counts']['vegetation'] / pixel_sums['vegetation'])

                if pixel_sums['not_vegetation'] == 0:
                    weights['not_vegetation'].append(0)
                else:
                    weights['not_vegetation'].append(image['stats']['counts']['not_vegetation'] / pixel_sums['not_vegetation'])

                if pixel_sums['unclassified'] == 0:
                    weights['unclassified'].append(0)
                else:
                    weights['unclassified'].append(image['stats']['counts']['unclassified'] / pixel_sums['unclassified'])         
        
            quintiles = {
                'B02': { 'vegetation': 0, 'not_vegetation': 0, 'unclassified': 0 },
                'B03': { 'vegetation': 0, 'not_vegetation': 0, 'unclassified': 0 },
                'B04': { 'vegetation': 0, 'not_vegetation': 0, 'unclassified': 0 },
                'B08': { 'vegetation': 0, 'not_vegetation': 0, 'unclassified': 0 },
            }

            for band in ['B02', 'B03', 'B04', 'B08']:
                quintiles[band]['vegetation'] = weighted_quantile(mean_values[band]['vegetation'], match_quintile, weights['vegetation'])
                quintiles[band]['not_vegetation'] = weighted_quantile(mean_values[band]['not_vegetation'], match_quintile, weights['not_vegetation'])
                quintiles[band]['unclassified'] = weighted_quantile(mean_values[band]['unclassified'], match_quintile, weights['unclassified'])

            for i in processed_images_indices:
                for band in ['B02', 'B03', 'B04', 'B08']:
                    if pixel_sums['valid_ground_pixels'] == 0:
                        metadata[i]['stats'][band]['scale'] = 1
                    else:
                        metadata[i]['stats'][band]['scale'] = sum([
                            ratios['vegetation'] * (quintiles[band]['vegetation'] / metadata[i]['stats'][band]['vegetation']),
                            ratios['not_vegetation'] * (quintiles[band]['not_vegetation'] / metadata[i]['stats'][band]['not_vegetation']),
                            ratios['unclassified'] * (quintiles[band]['unclassified'] / metadata[i]['stats'][band]['unclassified']),
                        ])

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


    array_to_raster(scl.astype('uint8'), reference_raster=best_image['path']['10m']['B08'], out_raster=os.path.join(out_dir, f"slc_{out_name}.tif"), dst_projection=dst_projection)

    bands_to_output = ['B02', 'B03', 'B04', 'B08']
    print('Merging band data..')
    for band in bands_to_output:
        print(f'Writing: {band}..')
        base_image = raster_to_array(metadata[0]['path']['10m'][band]).astype('float32')

        for i in processed_images_indices:
            if i == 0:
                if match_mean and len(processed_images_indices) > 1:
                    base_image = base_image * metadata[i]['stats'][band]['scale']

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

        array_to_raster(np.ma.masked_where(scl == 0, base_image).astype('uint16'), reference_raster=best_image['path']['10m'][band], out_raster=os.path.join(out_dir, f"{band}_{out_name}.tif"), dst_projection=dst_projection)

    print(f'Completed mosaic in: {round((time() - start_time) / 60, 1)}m')
