import sys; sys.path.append('..'); sys.path.append('../lib/')
from lib.raster_io import raster_to_array, array_to_raster
from lib.raster_resample import resample
from lib.stats_filters import mode_filter
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
        image_metadata['path'] = list_of_SAFE_images[index]
        metadata.append(image_metadata)

    lowest_invalid_percentage = 100
    best_image = None

    # Select the image with the most valid pixels.
    for meta in metadata:
        if meta['ALL_INVALID_PERCENTAGE'] < lowest_invalid_percentage:
            best_image = meta
            lowest_invalid_percentage = meta['ALL_INVALID_PERCENTAGE']
    
    # Calculate the time difference from each image to the best image
    for meta in metadata:
        meta['time_difference'] = abs(meta['timestamp'] - best_image['timestamp'])

    # Sort by distance to best_image
    metadata = sorted(metadata, key=lambda k: k['time_difference']) 
    
    return metadata


def mosaic_tile(list_of_SAFE_images, out_dir, out_name='mosaic', cutoff_percentage=1, cloud_cutoff=2, border_cut=101):

    # Verify input
    assert isinstance(list_of_SAFE_images, list), "Input is not a list. [path_to_safe_file1, path_to_safe_file2, ...]"
    assert isinstance(out_dir, str), f"Outdir is not a string: {out_dir}"
    assert isinstance(out_name, str), f"out_name is not a string: {out_name}"
    assert len(list_of_SAFE_images) > 1, "List is empty or only a single image."

    metadata = prepare_metadata(list_of_SAFE_images)
    
    best_image = metadata[0]
    best_image_bands = get_band_paths(best_image['path'])
    
    print('Resampling and reading base image.')
    
    # Resample the tracking landcover array
    mask = raster_to_array(resample(best_image_bands['20m']['SCL'], reference_raster=best_image_bands['10m']['B04'])).astype('uint8')
    nodata = cv2.dilate(np.where(mask == 0, 1, 0).astype('uint8'), np.ones((border_cut, border_cut)), iterations=1).astype('bool')
    cldprb = (raster_to_array(resample(best_image_bands['QI']['CLDPRB'], reference_raster=best_image_bands['10m']['B04'])) > cloud_cutoff).astype('bool')

    mask = np.ma.array(mask, mask=((mask == 1) | (mask == 3) | (mask == 7))).mask
    mask = np.logical_or(mask, nodata)
    mask = np.logical_or(mask, cldprb)
    
    metadata[0]['valid_mask'] = mask

    tracking_array = np.zeros(mask.shape, dtype='uint8')
    
    coverage = best_image['ALL_INVALID_PERCENTAGE']
    current_image = 1  # The 0 index is for the best image
    # Loop the images and update the tracking array
    while coverage > cutoff_percentage and current_image < len(metadata) - 1:
        current_metadata = metadata[current_image]
        current_bands = get_band_paths(current_metadata['path'])
        ex_mask = raster_to_array(resample(current_bands['20m']['SCL'], reference_raster=current_bands['10m']['B04'])).astype('uint8')
        ex_nodata = cv2.dilate(np.where(ex_mask == 0, 1, 0).astype('uint8'), np.ones((border_cut, border_cut)), iterations=1).astype('bool')
        ex_cldprb = (raster_to_array(resample(current_bands['QI']['CLDPRB'], reference_raster=current_bands['10m']['B04'])) > cloud_cutoff).astype('bool')
        
        ex_mask = np.ma.array(ex_mask, mask=((ex_mask == 1) | (ex_mask == 3) | (ex_mask == 7))).mask
        ex_mask = np.logical_or(ex_mask, ex_nodata)
        ex_mask = np.logical_or(ex_mask, ex_cldprb)

        metadata[current_image]['valid_mask'] = ex_mask

        change_mask = (np.logical_and(mask == True, np.logical_and((mask != ex_mask), (ex_mask == False)))).astype('bool')
        
        # Check if change is > 1%, else skip
        if (change_mask.sum() / change_mask.size) >= 0.01:
            mask = np.logical_and(mask, ex_mask).astype('bool')
            
            # Add to tracking array
            tracking_array = tracking_array + (change_mask * current_image)

            # Update coverage
            coverage = (mask.sum() / mask.size) * 100

            print(f'Updating the tracking array: {round(coverage, 3)} towards goal: {cutoff_percentage}')
        else:
            print('Skipped an image.')

        current_image += 1  


    print('Creating draft B02 band')
    best_image_b2 = raster_to_array(best_image_bands['10m']['B02'])
    for index, image in enumerate(metadata):

        # Stop when no more processed images
        if index > (current_image - 1):
            break
        
        # Skip the base image
        if index == 0:
            continue
        
        paths = get_band_paths(image['path'])
        
        add_band = raster_to_array(paths['10m']['B02'])

        best_image_b2 = np.where(tracking_array == index, add_band, best_image_b2)
    

    print('Retesting B02')
    for index, image in enumerate(metadata):

        # Stop when no more processed images
        if index > (current_image - 1):
            break
        
        # Skip the base image
        if index == 0:
            continue
        
        paths = get_band_paths(image['path'])

        ex_b2_raw = raster_to_array(paths['10m']['B08'])
        ex_b2_masked = np.ma.array(ex_b2_raw, mask=image['valid_mask'])
        ex_b2_mean = ex_b2_masked.mean()

        base_b2 = np.ma.array(best_image_b2, mask=image['valid_mask'])
        base_b2_mean = base_b2.mean()
        
        # Only update if image is atleast 5%. Is between 500 and 3000 and mean lower that base.
        if ((image['valid_mask'] == False).sum() / image['valid_mask'].size) > 0.05:
            if ex_b2_mean > 500 and ex_b2_mean < 3000:
                if (ex_b2_mean / base_b2_mean) < 1 - (index / 20):
                    tracking_array = np.where(image['valid_mask'] == False, index, tracking_array)
                    base_b2 = np.where(image['valid_mask'] == False, ex_b2_raw, base_b2)


    print('Filtering the tracking array.')
    # Run a mode filter on the tracking array
    tracking_array = mode_filter(tracking_array, 15)
    tracking_array = mode_filter(tracking_array, 5, 3)
    array_to_raster(tracking_array, reference_raster=best_image_bands['10m']['B08'], out_raster=f"{out_dir}/tracking_{out_name}.tif")


    print('Merging band data.')
    for band in ['B02', 'B03', 'B04', 'B08']:
        base_image = raster_to_array(best_image_bands['10m'][band])
        
        for index, image in enumerate(metadata):

            # Stop when no more processed images
            if index > (current_image - 1):
                break
            
            # Skip the base image
            if index == 0:
                continue
            
            if (tracking_array == index).sum() == 0:
                continue
            
            paths = get_band_paths(image['path'])
            
            add_band = raster_to_array(paths['10m'][band])

            base_image = np.where(tracking_array == np.full(tracking_array.shape, index), add_band, base_image)        
            
        array_to_raster(base_image, reference_raster=best_image_bands['10m'][band], out_raster=f"{out_dir}/{band}_{out_name}.tif")


if __name__ == "__main__":
    folder = "/mnt/c/Users/caspe/Desktop/tmp/"
    out_folder = "/mnt/c/Users/caspe/Desktop/tmp/mosaic"
    images = glob(folder + "*.*")
    
    mosaic_tile(images, out_folder, "mosaic", cutoff_percentage=2, cloud_cutoff=1)

