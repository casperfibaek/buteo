import sys; sys.path.append('..'); sys.path.append('../lib/')
from lib.raster_io import raster_to_array, array_to_raster
from lib.raster_resample import resample
from lib.stats_filters import mode_filter, feather_s2_filter
from lib.stats_kernel import create_kernel
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
        },
        "QI": {
            'CLDPRB': None
        }
    }
    
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
    
    # Did we get all the bands?
    for resolution in bands:
        for name in bands[resolution]:
            assert bands[resolution][name] != None, f'Could not find band: {resolution, name}'
    
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
        + metadata["UNCLASSIFIED_PERCENTAGE"]
        + metadata["MEDIUM_PROBA_CLOUDS_PERCENTAGE"]
        + metadata["HIGH_PROBA_CLOUDS_PERCENTAGE"]
        + metadata["THIN_CIRRUS_PERCENTAGE"]
        + metadata["SNOW_ICE_PERCENTAGE"]
    )
    
    metadata["ALL_INVALID_PERCENTAGE"] = (
        metadata["INVALID_PERCENTAGE"] + metadata['NODATA_PIXEL_PERCENTAGE']
    )
    
    metadata["timestamp"] = float(metadata['DATATAKE_SENSING_START'].timestamp())
    
    metadata["stats"] = {
        "B02": {
            "scaling": 1,
            "vegetation": None,
            "non_vegetation": None,
        },
        "B03": {
            "scaling": 1,
            "vegetation": None,
            "non_vegetation": None,
        },
        "B04": {
            "scaling": 1,
            "vegetation": None,
            "non_vegetation": None,
        },
        "B08": {
            "scaling": 1,
            "vegetation": None,
            "non_vegetation": None,
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

    # Sort by distance to best_image
    sorted_by_valid = sorted(metadata, key=lambda k: k['ALL_INVALID_PERCENTAGE'])
    
    # Select the image with the most valid pixels.
    for meta in metadata:
        if meta['ALL_INVALID_PERCENTAGE'] < lowest_invalid_percentage:
            lowest_invalid_percentage = meta['ALL_INVALID_PERCENTAGE']
            best_image = meta

    # Calculate the time difference from each image to the best image
    for meta in metadata:
        meta['time_difference'] = abs(meta['timestamp'] - best_image['timestamp'])

    # Sort by distance to best_image
    metadata = sorted(metadata, key=lambda k: k['time_difference']) 
    
    return metadata


# TODO: add proj4 string support.

def mosaic_tile(list_of_SAFE_images, out_dir, out_name='mosaic', feather=True, cutoff_percentage=2, cloud_cutoff=2, border_cut=61, invalid_contract=5, invalid_expand=41, feather_dist=41, match_mean=True, match_quintile=0.25):

    # Verify input
    assert isinstance(list_of_SAFE_images, list), "Input is not a list. [path_to_safe_file1, path_to_safe_file2, ...]"
    assert isinstance(out_dir, str), f"Outdir is not a string: {out_dir}"
    assert isinstance(out_name, str), f"out_name is not a string: {out_name}"
    assert len(list_of_SAFE_images) > 1, "List is empty or only a single image."

    metadata = prepare_metadata(list_of_SAFE_images)
    
    # Sorted by best, so 0 is the best one.
    best_image = metadata[0]
    
    print('Resampling and reading base image.')
    border_kernel = create_kernel(border_cut, weighted_edges=False, weighted_distance=False, normalise=False).astype('uint8')
    invalid_kernel = create_kernel(invalid_expand, weighted_edges=False, weighted_distance=False, normalise=False).astype('uint8')
    invalid_kernel_small = create_kernel(invalid_contract, weighted_edges=False, weighted_distance=False, normalise=False).astype('uint8')
    
    # Resample the tracking landcover array
    slc = raster_to_array(resample(best_image['path']['20m']['SCL'], reference_raster=best_image['path']['10m']['B04'])).astype('uint8')
    nodata = cv2.dilate(np.where(slc == 0, 1, 0).astype('uint8'), border_kernel, iterations=1).astype('bool')
    cldprb = (raster_to_array(resample(best_image['path']['QI']['CLDPRB'], reference_raster=best_image['path']['10m']['B04'])) > cloud_cutoff).astype('bool')

    mask = np.ma.array(slc, mask=((slc == 1) | (slc == 3))).mask
    mask = np.logical_or(mask, nodata)
    mask = np.logical_or(mask, cldprb)
    mask = cv2.erode(mask.astype('uint8'), invalid_kernel_small)
    mask = cv2.dilate(mask.astype('uint8'), invalid_kernel).astype('bool')
    
    vegetation_means = {
        "B02": [],
        "B03": [],
        "B04": [],
        "B08": [],
    }

    non_vegetation_means = {
        "B02": [],
        "B03": [],
        "B04": [],
        "B08": [],
    }
    
    if match_mean:
        vegetation_mask = np.logical_or(np.ma.masked_equal(slc != 4, slc).data, mask)
        non_vegetation_mask = np.logical_or(np.ma.masked_equal(slc != 5, slc).data, mask)

        for band in ['B02', 'B03', 'B04', 'B08']:
            base_ref = raster_to_array(metadata[0]['path']['10m'][band])
            metadata[0]['stats'][band]['vegetation'] = np.ma.array(base_ref, mask=vegetation_mask).mean()
            metadata[0]['stats'][band]['non_vegetation'] = np.ma.array(base_ref, mask=non_vegetation_mask).mean()
        
            vegetation_means[band].append(metadata[0]['stats'][band]['vegetation'])
            non_vegetation_means[band].append(metadata[0]['stats'][band]['non_vegetation'])

        vegetation_mask = None
        non_vegetation_mask = None

    # Clear memory
    nodata = None
    cldprb = None
    
    metadata[0]['valid_mask'] = mask

    tracking_array = np.zeros(mask.shape, dtype='uint8')
      
    coverage = (mask.sum() / mask.size) * 100
    current_image_index = 1  # The 0 index is for the best image
    
    processed_images_indices = [0]
    
    print(f'Initial. tracking array: {round(coverage, 3)} towards goal: {cutoff_percentage}')
    # Loop the images and update the tracking array
    while coverage > cutoff_percentage and current_image_index < len(metadata) - 1:
        current_metadata = metadata[current_image_index]
        ex_slc = raster_to_array(resample(current_metadata['path']['20m']['SCL'], reference_raster=current_metadata['path']['10m']['B04'])).astype('uint8')
        ex_nodata = cv2.dilate(np.where(ex_slc == 0, 1, 0).astype('uint8'), border_kernel, iterations=1).astype('bool')
        ex_cldprb = (raster_to_array(resample(current_metadata['path']['QI']['CLDPRB'], reference_raster=current_metadata['path']['10m']['B04'])) > cloud_cutoff).astype('bool')
        
        ex_mask = np.ma.array(ex_slc, mask=((ex_slc == 1) | (ex_slc == 3))).mask
        ex_mask = np.logical_or(ex_mask, ex_nodata)
        ex_mask = np.logical_or(ex_mask, ex_cldprb)
        ex_mask = cv2.erode(ex_mask.astype('uint8'), invalid_kernel_small).astype('bool')
        ex_mask = cv2.dilate(ex_mask.astype('uint8'), invalid_kernel).astype('bool')
        
        change_mask = (np.logical_and(mask == True, np.logical_and((mask != ex_mask), (ex_mask == False)))).astype('bool')
        
        # Only process if change is more that 0.5 procent.
        if ((change_mask.sum() / change_mask.size) * 100) > 0.5:
            if match_mean:
                ex_vegetation_mask = np.logical_or(np.ma.masked_equal(ex_slc != 4, ex_slc).data, change_mask == False)
                ex_non_vegetation_mask = np.logical_or(np.ma.masked_equal(ex_slc != 5, ex_slc).data, change_mask == False)
                
                for band in ['B02', 'B03', 'B04', 'B08']:
                    base_ref = raster_to_array(metadata[0]['path']['10m'][band])
                
                    ex_ref = raster_to_array(metadata[current_image_index]['path']['10m'][band])

                    metadata[current_image_index]['stats'][band]['vegetation'] = np.ma.array(ex_ref, mask=ex_vegetation_mask).mean()
                    metadata[current_image_index]['stats'][band]['non_vegetation'] = np.ma.array(ex_ref, mask=ex_non_vegetation_mask).mean()
                    
                    # Set them to the last valid scale, if for some reason the means are invalid even with the 0.5% filter.
                    if metadata[current_image_index]['stats'][band]['vegetation'] == 0 or metadata[current_image_index]['stats'][band]['non_vegetation'] == 0:
                        print('Warning: mean was zero for changes array.. This should really not happen.')
                        metadata[current_image_index]['stats'][band]['vegetation'] = metadata[processed_images_indices[len(processed_images_indices) - 1]]['stats'][band]['vegetation']
                        metadata[current_image_index]['stats'][band]['non_vegetation'] = metadata[processed_images_indices[len(processed_images_indices) - 1]]['stats'][band]['non_vegetation']
                    
                    vegetation_means[band].append(metadata[current_image_index]['stats'][band]['vegetation'])
                    non_vegetation_means[band].append(metadata[current_image_index]['stats'][band]['non_vegetation'])
            
                ex_vegetation_mask = None
                ex_non_vegetation_mask = None

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

            print(f'Updating tracking array: {round(coverage, 3)} towards goal: {cutoff_percentage}')
        else:
            print('Skipping image due to low change.. (0.5% threshold)')

        current_image_index += 1      

    # Calculate the scaling factors
    if match_mean:
        for band in ['B02', 'B03', 'B04', 'B08']:
            vegetation_quantile = np.quantile(vegetation_means[band], match_quintile)
            non_vegetation_quantile = np.quantile(non_vegetation_means[band], match_quintile)

            for i in processed_images_indices:
                metadata[i]['stats'][band]['scaling'] = ((vegetation_quantile / (metadata[i]['stats'][band]['vegetation']) + (non_vegetation_quantile /  metadata[i]['stats'][band]['non_vegetation'])) / 2)


    print('Filtering tracking array.')
    # Run a mode filter on the tracking array
    tracking_array = mode_filter(tracking_array, 9)
    tracking_array = mode_filter(tracking_array, 5, 2)
    array_to_raster(tracking_array, reference_raster=best_image['path']['10m']['B08'], out_raster=f"{out_dir}/tracking_{out_name}.tif")

    # Prepare and save feathers (takes alot of ram... revise?)
    if feather:
        print('Precalculating feathers..')
        feathers = {}
        for i in processed_images_indices:
            feathers[f"{i}"] = feather_s2_filter(tracking_array, i, feather_dist).astype('float32')

    print('Merging band data.')
    for band in ['B02', 'B03', 'B04', 'B08']:
        base_image = raster_to_array(best_image['path']['10m'][band]).astype('float32')

        for i in processed_images_indices:
            if i == 0:
                if match_mean:
                    base_image = base_image * best_image['stats'][band]['scaling']

                if feather is True:
                    base_image = base_image * feathers[f"{i}"]

            else:
                add_band = raster_to_array(metadata[i]['path']['10m'][band]).astype('float32')
                
                if match_mean:
                    add_band = add_band * metadata[i]['stats'][band]['scaling']
            
                if feather is True:
                    base_image = np.add(base_image, (add_band * feathers[f"{i}"]))
                else:
                    base_image = np.where(tracking_array == np.full(tracking_array.shape, i).astype('uint8'), add_band, base_image).astype('float32')
            
        array_to_raster(base_image.astype('uint16'), reference_raster=best_image['path']['10m'][band], out_raster=f"{out_dir}/{band}_{out_name}.tif")


if __name__ == "__main__":
    folder = "/mnt/c/Users/caspe/Desktop/tmp/"
    out_dir = "/mnt/c/Users/caspe/Desktop/tmp/mosaic"
    images = glob(folder + "*.*")
    
    mosaic_tile(
        images,
        out_dir,
        "mosaic",
        feather=True,
        cutoff_percentage=2,
        cloud_cutoff=2,
        border_cut=51,
        invalid_contract=3,
        invalid_expand=51,
        feather_dist=51,
        match_mean=True,
        match_quintile=0.25,
    )

