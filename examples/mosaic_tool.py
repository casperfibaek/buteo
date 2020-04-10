import sys; sys.path.append('..'); sys.path.append('../lib/')
from lib.raster_io import raster_to_array, array_to_raster, raster_to_metadata
from lib.raster_resample import resample
from lib.stats_filters import mode_filter, feather_s2_filter, mean_filter, median_filter
from lib.stats_local_no_kernel import radiometric_quality
from lib.stats_kernel import create_kernel
from lib.utils_core import weighted_quantile, madstd
from time import time
import cv2
import os
import xml.etree.ElementTree as ET
import datetime
from shutil import copyfile
import math
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
            'CLDPRB_20m': None,
            'CLDPRB_60m': None,
        }
    }
    
    assert os.path.isdir(safe_folder), f"Could not find folder: {safe_folder}"
    
    bands['QI']['CLDPRB_20m'] = glob(f"{safe_folder}/GRANULE/*/QI_DATA/MSK_CLDPRB_20m.jp2")[0]
    bands['QI']['CLDPRB_60m'] = glob(f"{safe_folder}/GRANULE/*/QI_DATA/MSK_CLDPRB_60m.jp2")[0]
    
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
        if band_name == 'AOT':
            bands['10m']['AOT'] = band

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
        if band_name == 'AOT':
            bands['20m']['AOT'] = band

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
        if band_name == 'AOT':
            bands['60m']['AOT'] = band
    
    for outer_key in bands:
        for inner_key in bands[outer_key]:
            current_band = bands[outer_key][inner_key]
            assert current_band != None, f'{outer_key} - {inner_key} was not found. Verify the folders. Was the decompression interrupted?'

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
        "ZENITH_ANGLE": None,
        "AZIMUTH_ANGLE": None,
        "SUN_ELEVATION": None,
        "folder": safe_folder,
        "gains": {}
    }

    meta_xml = os.path.join(safe_folder, "MTD_MSIL2A.xml")
    meta_solar = glob(safe_folder + '/GRANULE/*/MTD_TL.xml')[0]

    assert os.path.isfile(meta_xml), f"{safe_folder} did not contain a valid metadata file."
    assert os.path.isfile(meta_solar), f"{meta_solar} did not contain a valid metadata file."

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
        if elem.tag == 'PHYSICAL_GAINS':
            if elem.attrib['bandId'] == '0':
                metadata['gains']['B01'] = float(elem.text)
            if elem.attrib['bandId'] == '1':
                metadata['gains']['B02'] = float(elem.text)
            if elem.attrib['bandId'] == '2':
                metadata['gains']['B03'] = float(elem.text)
            if elem.attrib['bandId'] == '3':
                metadata['gains']['B04'] = float(elem.text)
            if elem.attrib['bandId'] == '4':
                metadata['gains']['B05'] = float(elem.text)
            if elem.attrib['bandId'] == '5':
                metadata['gains']['B06'] = float(elem.text)
            if elem.attrib['bandId'] == '6':
                metadata['gains']['B07'] = float(elem.text)
            if elem.attrib['bandId'] == '7':
                metadata['gains']['B08'] = float(elem.text)
            if elem.attrib['bandId'] == '8':
                metadata['gains']['B8A'] = float(elem.text)
            if elem.attrib['bandId'] == '9':
                metadata['gains']['B09'] = float(elem.text)
            if elem.attrib['bandId'] == '10':
                metadata['gains']['B10'] = float(elem.text)
            if elem.attrib['bandId'] == '11':
                metadata['gains']['B11'] = float(elem.text)
            if elem.attrib['bandId'] == '12':
                metadata['gains']['B12'] = float(elem.text)

    # Parse the xml tree and add metadata
    root = ET.parse(meta_solar).getroot()
    for elem in root.iter():
        if elem.tag == 'Mean_Sun_Angle':
            metadata['ZENITH_ANGLE'] = float(elem.find('ZENITH_ANGLE').text)
            metadata['SUN_ELEVATION'] = 90 - metadata['ZENITH_ANGLE']
            metadata['AZIMUTH_ANGLE'] = float(elem.find('AZIMUTH_ANGLE').text)

    # Did we get all the metadata?
    for name in metadata:
        assert (
            metadata[name] != None
        ), f"Input metatadata file invalid. {metadata[name]}"

    metadata["INVALID"] = (
        metadata['NODATA_PIXEL_PERCENTAGE']
        + metadata["SATURATED_DEFECTIVE_PIXEL_PERCENTAGE"]
        + metadata["CLOUD_SHADOW_PERCENTAGE"]
        + metadata["MEDIUM_PROBA_CLOUDS_PERCENTAGE"]
        + metadata["HIGH_PROBA_CLOUDS_PERCENTAGE"]
        + metadata["THIN_CIRRUS_PERCENTAGE"]
        + metadata["SNOW_ICE_PERCENTAGE"]
        + metadata["DARK_FEATURES_PERCENTAGE"]
    )
    
    metadata["timestamp"] = float(metadata['DATATAKE_SENSING_START'].timestamp())

    return metadata

def get_time_difference(dict):
    return dict['time_difference']

def assess_radiometric_quality(metadata, calc_quality='high', score=False):
    if calc_quality == 'high':
        scl = raster_to_array(metadata['path']['20m']['SCL']).astype('intc')
        aot = raster_to_array(metadata['path']['20m']['AOT']).astype('intc')
        band_02 = raster_to_array(metadata['path']['20m']['B02']).astype('intc')
        band_12 = raster_to_array(metadata['path']['20m']['B12']).astype('intc')
        band_cldprb = raster_to_array(metadata['path']['QI']['CLDPRB_20m'])
        distance = 101
    else:
        scl = raster_to_array(metadata['path']['60m']['SCL']).astype('intc')
        aot = raster_to_array(metadata['path']['60m']['AOT']).astype('intc')
        band_cldprb = raster_to_array(metadata['path']['QI']['CLDPRB_60m'])
        band_02 = raster_to_array(metadata['path']['60m']['B02']).astype('intc')
        band_12 = raster_to_array(metadata['path']['60m']['B12']).astype('intc')
        distance = 33

    kernel_nodata = create_kernel(201, weighted_edges=False, weighted_distance=False, normalise=False).astype('uint8')
    
    # Dilate nodata values by 1km each side 
    nodata_dilated = cv2.dilate((scl == 0).astype('uint8'), kernel_nodata).astype('intc')

    darkprb = np.zeros(scl.shape)
    darkprb = np.where(scl == 2, 55, 0)
    darkprb = np.where(scl == 3, 45, darkprb).astype('uint8')
    darkprb = cv2.GaussianBlur(darkprb, (distance, distance), 0).astype(np.double)
    band_cldprb = cv2.GaussianBlur(band_cldprb, (distance, distance), 0).astype(np.double)
    
    quality = np.zeros(scl.shape, dtype=np.double)
    
    td = 0.0 if score is True else metadata['time_difference'] / 86400

    # OBS: the radiometric_quality functions mutates the quality input.
    combined_score = radiometric_quality(scl, band_02, band_12, band_cldprb, darkprb, aot, nodata_dilated, quality, td, metadata['SUN_ELEVATION'])
    
    if score is True:
        return combined_score
    
    blur_dist = int(math.ceil(distance / 2))
    quality_blurred = cv2.GaussianBlur(quality, (blur_dist, blur_dist), 0).astype(np.double)
    
    return quality_blurred, scl, band_cldprb

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
        image_metadata['name'] = os.path.basename(os.path.normpath(image_metadata['folder'])).split('_')[-1].split('.')[0]
        metadata.append(image_metadata)

    # lowest_invalid_percentage = 100
    best_image = None
    highest_quality = 0
    
    # best_images_names = []
    # best_images = []
    
    # Find the image with the lowest invalid percentage
    for index, value in enumerate(metadata):
        quality_score = assess_radiometric_quality(value, calc_quality='low', score=True)
        metadata[index]['quality_score'] = quality_score
        if quality_score > highest_quality:
            highest_quality = quality_score
            best_image = value

    # for meta in metadata:
    #     if (meta['INVALID'] - lowest_invalid_percentage) <= 10:
    #         if meta['name'] not in best_images_names:
    #             best_images.append(meta)
    #             best_images_names.append(meta['name'])

    # if len(best_images) != 1:
    #     # Search the top 10% images for the best image
    #     max_quality_score = 0
    #     for image in best_images:
    #         quality_score = assess_radiometric_quality(image, calc_quality='low', score=True)

    #         if quality_score > max_quality_score:
    #             max_quality_score = quality_score
    #             best_image = image

    # Calculate the time difference from each image to the best image
    for meta in metadata:
        meta['time_difference'] = abs(meta['timestamp'] - best_image['timestamp'])

    # Sort by distance to best_image
    metadata = sorted(metadata, key=lambda k: -k['quality_score'])
    
    # import pdb; pdb.set_trace()
    
    return metadata

# TODO: Read more metadata: 
# Zenith array, read gain and spectral information to use in harmonisation step.

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
    ideal_percent=100,
    feather_dist=31,
    filter_tracking=True,
    match_mean=True,
    max_days=60,
    max_images_include=15,
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
    best_image_name = best_image['name']

    print(f'Selected: {best_image_name}')

    print('Resampling and reading base image..')
    quality, scl, cldprb = assess_radiometric_quality(best_image)
    tracking_array = np.zeros(quality.shape, dtype='uint8')
    
    if match_mean is True:
        metadata[0]['scl'] = np.copy(scl)
        metadata[0]['quality'] = quality
  
    time_limit = (max_days * 86400)

    avg_quality = (quality.sum() / quality.size)
    i = 1  # The 0 index is for the best image
    processed_images_indices = [0]

    print(f'Initial. tracking array: (quality {round(avg_quality, 2)}%) (0/{max_days} days) (goal {ideal_percent}%)')
    # Loop the images and update the tracking array
    while (
        (avg_quality < ideal_percent)
        and i < len(metadata) - 1
        and len(processed_images_indices) <= max_images_include
    ):
        if (metadata[i]['time_difference'] > time_limit):
            i += 1
            continue

        # Time difference
        td = int(round(metadata[i]['time_difference'] / 86400, 0))  

        ex_quality, ex_scl, cldprb = assess_radiometric_quality(metadata[i])

        change_mask = ex_quality > quality

        ex_quality_test = np.where(change_mask, ex_quality, quality)
        ex_quality_avg = (ex_quality_test.sum() / ex_quality_test.size)

        # Linear decrease of threshold( 0 = 0.25% & 100 = 1.75%)
        threshold = -0.015 * avg_quality + 1.75

        if (ex_quality_avg - avg_quality) > threshold:
            
            # Udpdate the trackers
            tracking_array = np.where(change_mask, i, tracking_array).astype('uint8')
            scl = np.where(change_mask, ex_scl, scl).astype('intc')

            if match_mean is True:
                metadata[i]['scl'] = ex_scl

            quality = ex_quality_test.astype(np.double)

            avg_quality = (quality.sum() / quality.size)

            processed_images_indices.append(i)

            img_name = metadata[i]['name']
            print(f'Updating tracking array: (quality {round(avg_quality, 2)}%) ({td}/{max_days} days) (goal {ideal_percent}%) (name {img_name})')
        else:
            print(f'Skipping image due to low change.. ({round(threshold, 3)}% threshold) ({td}/{max_days} days)')

        i += 1

    # Free memory
    quality = None
    ex_scl = None
    ex_quality = None
    ex_quality_test = None
    
    multiple_images = len(processed_images_indices) > 1

    # Only merge images if there are more than one.
    if match_mean is True and multiple_images is True:

        print('Harmonising layers..')
        
        total_counts = 0
        counts = []
        weights = []
        gains = { 'B02': [], 'B03': [], 'B04': [], 'B08': [] }
        target_gain = { 'B02': 0, 'B03': 0, 'B04': 0, 'B08': 0 }
        ratios = { 'B02': [], 'B03': [], 'B04': [], 'B08': [] }
        
        for i in processed_images_indices:
            metadata[i]['stats'] = { 'B02': {}, 'B03': {}, 'B04': {}, 'B08': {} }
            pixel_count = (tracking_array == i).sum()
            total_counts += pixel_count
            counts.append(pixel_count)
            
            for band in ['B02', 'B03', 'B04', 'B08']:
                gains[band].append(metadata[i]['gains'][band])
        
        for i in range(len(processed_images_indices)):
            w = counts[i] / total_counts
            
            for band in ['B02', 'B03', 'B04', 'B08']:
                target_gain[band] += gains[band][i] * w

            weights.append(w)
        
        for i in range(len(processed_images_indices)):
            for band in ['B02', 'B03', 'B04', 'B08']:
                ratios[band].append(target_gain[band] / metadata[processed_images_indices[i]]['gains'][band])

        medians = { 'B02': [], 'B03': [], 'B04': [], 'B08': [] }
        madstds = { 'B02': [], 'B03': [], 'B04': [], 'B08': [] }

        for v, i in enumerate(processed_images_indices):
            layer_mask = ((metadata[i]['scl'] == 4) | (metadata[i]['scl'] == 5)) == False
            
            if layer_mask.sum() / tracking_array.size <= 0.02:
                layer_mask = ((metadata[i]['scl'] == 4) | (metadata[i]['scl'] == 5) | (metadata[i]['scl'] == 7)) == False

            
            if layer_mask.sum() / tracking_array.size <= 0.02:
                layer_mask = ((metadata[i]['scl'] == 4) | (metadata[i]['scl'] == 5) | (metadata[i]['scl'] == 6) | (metadata[i]['scl'] == 7)) == False
            
            if layer_mask.sum() / tracking_array.size <= 0.02:
                layer_mask = metadata[i]['scl'] == 0

            for band in ['B02', 'B03', 'B04', 'B08']:
                if band == 'B08':
                    array = raster_to_array(resample(metadata[i]['path']['10m'][band], reference_raster=metadata[i]['path']['20m']['B02']))
                else:
                    array = raster_to_array(metadata[i]['path']['20m'][band])
                    
                array = array * (metadata[i]['gains'][band] / target_gain[band])

                calc_array = np.ma.array(array, mask=layer_mask)
                med, mad = madstd(calc_array)

                if med == 0 or mad == 0:
                    med, mad = madstd(array)
                    
                medians[band].append(med)
                madstds[band].append(mad)
        
        targets_median = { 'B02': None, 'B03': None, 'B04': None, 'B08': None }
        targets_madstd = { 'B02': None, 'B03': None, 'B04': None, 'B08': None }
        
        for band in ['B02', 'B03', 'B04', 'B08']:
            targets_median[band] = np.average(medians[band], weights=weights)
            targets_madstd[band] = np.average(madstds[band], weights=weights)
    
        for v, i in enumerate(processed_images_indices):
            for band in ['B02', 'B03', 'B04', 'B08']:
                metadata[i]['stats'][band]['src_median'] = medians[band][v] if medians[band][v] > 0 else targets_median[band]
                metadata[i]['stats'][band]['src_madstd'] = madstds[band][v] if madstds[band][v] > 0 else targets_madstd[band]
                metadata[i]['stats'][band]['target_median'] = targets_median[band]
                metadata[i]['stats'][band]['target_madstd'] = targets_madstd[band]


    # Resample scl and tracking array
    tracking_array = raster_to_array(resample(array_to_raster(tracking_array, reference_raster=best_image['path']['20m']['B04']), reference_raster=best_image['path']['10m']['B04']))
    scl = raster_to_array(resample(array_to_raster(scl, reference_raster=best_image['path']['20m']['B04']), reference_raster=best_image['path']['10m']['B04']))

    if filter_tracking is True and multiple_images is True:
        print('Filtering tracking array..')
        # Run a mode filter on the tracking arrayclear
        tracking_array = mode_filter(tracking_array, 7).astype('uint8')
    
    array_to_raster(tracking_array.astype('uint8'), reference_raster=best_image['path']['10m']['B08'], out_raster=os.path.join(out_dir, f"tracking_{out_name}.tif"), dst_projection=dst_projection)

    if feather and multiple_images is True:
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

        count = 0
        for i in processed_images_indices:

            if match_mean and len(processed_images_indices) > 1:
                src_med = metadata[i]['stats'][band]['src_median']
                src_mad = metadata[i]['stats'][band]['src_madstd']
                target_med = metadata[i]['stats'][band]['target_median']
                target_mad = metadata[i]['stats'][band]['target_madstd']

            if i == 0:
                if match_mean and len(processed_images_indices) > 1:
                    dif = ((base_image * ratios[band][count]) - src_med)
                    base_image = ((dif * target_mad) / src_mad) + target_med
                    base_image = np.where(base_image >= 0, base_image, 0)
                    
                if feather is True and len(processed_images_indices) > 1:
                    base_image = base_image * feathers[str(i)]

            else:
                add_band = raster_to_array(metadata[i]['path']['10m'][band]).astype('float32')
                
                if match_mean:                    
                    dif = ((add_band * ratios[band][count]) - src_med)
                    add_band = ((dif * target_mad) / src_mad) + target_med
                    add_band = np.where(add_band >= 0, add_band, 0)

                if feather is True:
                    base_image = np.add(base_image, (add_band * feathers[str(i)]))
                else:
                    base_image = np.where(tracking_array == i, add_band, base_image).astype('float32')
            
            count += 1

        array_to_raster(np.rint(base_image).astype('uint16'), reference_raster=best_image['path']['10m'][band], out_raster=os.path.join(out_dir, f"{band}_{out_name}.tif"), dst_projection=dst_projection)

    print(f'Completed mosaic in: {round((time() - start_time) / 60, 1)}m')
