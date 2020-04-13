import sys; sys.path.append('..'); sys.path.append('../lib/')
from lib.raster_io import raster_to_array, array_to_raster, raster_to_metadata
from lib.raster_resample import resample
from lib.stats_filters import mode_filter, feather_s2_filter
from lib.stats_local_no_kernel import radiometric_quality
from lib.stats_kernel import create_kernel
from lib.utils_core import madstd
from multiprocessing import Pool, cpu_count
from time import time
import cv2
import os
import xml.etree.ElementTree as ET
import datetime
from shutil import copyfile
import math
from glob import glob
import numpy as np

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


def assess_radiometric_quality(metadata, calc_quality='high', score=False):
    if calc_quality == 'high':
        scl = raster_to_array(metadata['path']['20m']['SCL']).astype('intc')
        aot = raster_to_array(metadata['path']['20m']['AOT']).astype('intc')
        band_02 = raster_to_array(metadata['path']['20m']['B02']).astype('intc')
        band_12 = raster_to_array(metadata['path']['20m']['B12']).astype('intc')
        band_cldprb = raster_to_array(metadata['path']['QI']['CLDPRB_20m'])
        distance = 63
    else:
        scl = raster_to_array(metadata['path']['60m']['SCL']).astype('intc')
        aot = raster_to_array(metadata['path']['60m']['AOT']).astype('intc')
        band_cldprb = raster_to_array(metadata['path']['QI']['CLDPRB_60m'])
        band_02 = raster_to_array(metadata['path']['60m']['B02']).astype('intc')
        band_12 = raster_to_array(metadata['path']['60m']['B12']).astype('intc')
        distance = 21

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
    
    return quality_blurred, scl


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

    # Find the image with the lowest invalid percentage
    pool_args = zip(metadata, ['low'] * len(metadata), [True] * len(metadata))
    pool = Pool(cpu_count())
    quality_scores = pool.starmap(assess_radiometric_quality, pool_args)
    pool.close()
    pool.join()
    
    for index, value in enumerate(metadata):
        metadata[index]['quality_score'] = quality_scores[index]
        if quality_scores[index] > highest_quality:
            highest_quality = quality_scores[index]
            best_image = value

    # Calculate the time difference from each image to the best image
    for meta in metadata:
        meta['time_difference'] = abs(meta['timestamp'] - best_image['timestamp'])

    # Sort by distance to best_image
    metadata = sorted(metadata, key=lambda k: -k['quality_score'])
    
   
    return metadata

# TODO: Add multiprocessing
# TODO: handle all bands
# TODO: add pansharpen
# TODO: ai resample of SWIR

def mosaic_tile(
    list_of_SAFE_images,
    out_dir,
    out_name='mosaic',
    dst_projection=None,
    feather=True,
    target_quality=100,
    threshold_change=0.5,
    threshold_quality=10.0,
    feather_dist=21,
    feather_scl=5,
    filter_tracking=True,
    match_mean=True,
    allow_nodata=False,
    max_days=60,
    max_images_include=15,
    max_images_search=25,
    output_scl=True,
    output_tracking=True,
    output_quality=False,
    verbose=True,
):
    start_time = time()

    # Verify input
    assert isinstance(list_of_SAFE_images, list), "list_of_SAFE_images is not a list. [path_to_safe_file1, path_to_safe_file2, ...]"
    assert isinstance(out_dir, str), f"out_dir is not a string: {out_dir}"
    assert isinstance(out_name, str), f"out_name is not a string: {out_name}"
    assert len(list_of_SAFE_images) > 1, "list_of_SAFE_images is empty or only a single image."

    if verbose: print('Selecting best image..')
    metadata = prepare_metadata(list_of_SAFE_images)
    
    # Sorted by best, so 0 is the best one.
    best_image = metadata[0]
    best_image_name = best_image['name']

    if verbose: print(f'Selected: {best_image_name} {out_name}')

    if verbose: print('Preparing base image..')
    master_quality, master_scl = assess_radiometric_quality(best_image)
    tracking_array = np.zeros(master_quality.shape, dtype='uint8')
    
    if match_mean is True:
        metadata[0]['scl'] = np.copy(master_scl)
  
    time_limit = (max_days * 86400)

    master_quality_avg = (master_quality.sum() / master_quality.size)
    i = 1  # The 0 index is for the best image
    processed_images_indices = [0]

    # Loop the images and update the tracking array (SYNTHESIS)
    if verbose: print(f'Initial. tracking array: (quality {round(master_quality_avg, 2)}%) (0/{max_days} days) (goal {target_quality}%)')
    while (
        (master_quality_avg < target_quality)
        and i < len(metadata) - 1
        and len(processed_images_indices) <= max_images_include
    ):
        if (metadata[i]['time_difference'] > time_limit):
            i += 1
            continue
        
        if (i >= max_images_search):
            if (master_scl == 0).sum() == 0 or allow_nodata is True:
                break
            if verbose: print('Continuing dispite reaching max_images_search as there is still nodata in tile..')

        # Time difference
        td = int(round(metadata[i]['time_difference'] / 86400, 0))  

        # Assess quality of current image
        quality, scl = assess_radiometric_quality(metadata[i])

        # Calculate changes. Always update nodata.
        change_mask = (quality > master_quality) | ((master_scl == 0) & (scl != 0))
        percent_change = (change_mask.sum() / change_mask.size) * 100
        
        # Calculate the global change in quality
        quality_global = np.where(change_mask, quality, master_quality)
        quality_global_avg = quality_global.sum() / quality_global.size
        quality_global_change = quality_global_avg - master_quality_avg
    
        if ((percent_change > threshold_change) and (quality_global_change > threshold_change)):
            
            # Udpdate the trackers
            tracking_array = np.where(change_mask, i, tracking_array).astype('uint8')
            master_scl = np.where(change_mask, scl, master_scl).astype('intc')
            master_quality = np.where(change_mask, quality, master_quality).astype(np.double)
            master_quality_avg = quality_global_avg

            # Save the scene classification in memory. This cost a bit of RAM but makes harmonisation much faster..
            metadata[i]['scl'] = scl.astype('uint8')

            # Append to the array that keeps track on which images are used in the synth process..
            processed_images_indices.append(i)

            img_name = metadata[i]['name']
            if verbose: print(f'Updating tracking array: (quality {round(master_quality_avg, 2)}%) ({td}/{max_days} days) (goal {target_quality}%) (name {img_name})')
        else:
            if verbose: print(f'Skipping image due to low change.. ({round(threshold_change, 3)}% threshold) ({td}/{max_days} days)')

        i += 1

    # Free memory
    change_mask = None
    change_mask_inv = None
    quality_global = None
    quality = None
    scl = None

    # Only merge images if there are more than one.
    multiple_images = len(processed_images_indices) > 1
    if match_mean is True and multiple_images is True:
        if verbose: print('Harmonising layers..')
        
        total_counts = 0
        counts = []
        weights = []
        
        for i in processed_images_indices:
            metadata[i]['stats'] = { 'B02': {}, 'B03': {}, 'B04': {}, 'B08': {} }
            pixel_count = (tracking_array == i).sum()
            total_counts += pixel_count
            counts.append(pixel_count)
        
        for i in range(len(processed_images_indices)):
            w = counts[i] / total_counts
            weights.append(w)

        medians = { 'B02': [], 'B03': [], 'B04': [], 'B08': [] }
        medians_4 = { 'B02': [], 'B03': [], 'B04': [], 'B08': [] }
        medians_5 = { 'B02': [], 'B03': [], 'B04': [], 'B08': [] }
        medians_6 = { 'B02': [], 'B03': [], 'B04': [], 'B08': [] }
        
        madstds = { 'B02': [], 'B03': [], 'B04': [], 'B08': [] }
        madstds_4 = { 'B02': [], 'B03': [], 'B04': [], 'B08': [] }
        madstds_5 = { 'B02': [], 'B03': [], 'B04': [], 'B08': [] }
        madstds_6 = { 'B02': [], 'B03': [], 'B04': [], 'B08': [] }

        for v, i in enumerate(processed_images_indices):
            layer_mask_4 = metadata[i]['scl'] != 4
            layer_mask_4_inv = layer_mask_4 == False
            layer_mask_5 = metadata[i]['scl'] != 5
            layer_mask_5_inv = layer_mask_5 == False
            layer_mask_6 = metadata[i]['scl'] != 6
            layer_mask_6_inv = layer_mask_6 == False

            layer_mask = (layer_mask_4 | layer_mask_5 | layer_mask_6 | (metadata[i]['scl'] == 7)) == False

            for band in ['B02', 'B03', 'B04', 'B08']:
                if band == 'B08':
                    array = raster_to_array(resample(metadata[i]['path']['10m'][band], reference_raster=metadata[i]['path']['20m']['B02']))
                else:
                    array = raster_to_array(metadata[i]['path']['20m'][band])

                calc_array = np.ma.array(array, mask=layer_mask)
                calc_array_4 = np.ma.array(array, mask=layer_mask_4)
                calc_array_5 = np.ma.array(array, mask=layer_mask_5)
                calc_array_6 = np.ma.array(array, mask=layer_mask_6)

                med, mad = madstd(calc_array)
                if layer_mask_4_inv.sum() > 1000:
                    med_4, mad_4 = madstd(calc_array_4)
                else:
                    med_4, mad_4 = madstd(calc_array)
                
                if layer_mask_5_inv.sum() > 1000:
                    med_5, mad_5 = madstd(calc_array_5)
                else:
                    med_5, mad_5 = madstd(calc_array)
                
                if layer_mask_6_inv.sum() > 1000:
                    med_6, mad_6 = madstd(calc_array_6)
                else:
                    med_6, mad_6 = madstd(calc_array)

                if med == 0 or mad == 0: med, mad = madstd(array)
                if med_4 == 0 or mad_4 == 0: med_4, mad_4 = (med, mad)
                if med_5 == 0 or mad_5 == 0: med_5, mad_5 = (med, mad)
                if med_6 == 0 or mad_6 == 0: med_6, mad_6 = (med, mad)
                    
                medians[band].append(med)
                medians_4[band].append(med_4)
                medians_5[band].append(med_5)
                medians_6[band].append(med_6)

                madstds[band].append(mad)
                madstds_4[band].append(mad_4)
                madstds_5[band].append(mad_5)
                madstds_6[band].append(mad_6)
        
        targets_median = { 'B02': None, 'B03': None, 'B04': None, 'B08': None }
        targets_median_4 = { 'B02': None, 'B03': None, 'B04': None, 'B08': None }
        targets_median_5 = { 'B02': None, 'B03': None, 'B04': None, 'B08': None }
        targets_median_6 = { 'B02': None, 'B03': None, 'B04': None, 'B08': None }

        targets_madstd = { 'B02': None, 'B03': None, 'B04': None, 'B08': None }
        targets_madstd_4 = { 'B02': None, 'B03': None, 'B04': None, 'B08': None }
        targets_madstd_5 = { 'B02': None, 'B03': None, 'B04': None, 'B08': None }
        targets_madstd_6 = { 'B02': None, 'B03': None, 'B04': None, 'B08': None }
        
        for band in ['B02', 'B03', 'B04', 'B08']:
            targets_median[band] = np.average(medians[band], weights=weights)
            targets_median_4[band] = np.average(medians_4[band], weights=weights)
            targets_median_5[band] = np.average(medians_5[band], weights=weights)
            targets_median_6[band] = np.average(medians_6[band], weights=weights)

            targets_madstd[band] = np.average(madstds[band], weights=weights)
            targets_madstd_4[band] = np.average(madstds_4[band], weights=weights)
            targets_madstd_5[band] = np.average(madstds_5[band], weights=weights)
            targets_madstd_6[band] = np.average(madstds_6[band], weights=weights)
    
        for v, i in enumerate(processed_images_indices):
            for band in ['B02', 'B03', 'B04', 'B08']:
                metadata[i]['stats'][band]['src_median'] = medians[band][v] if medians[band][v] > 0 else targets_median[band]
                metadata[i]['stats'][band]['src_median_4'] = medians_4[band][v] if medians_4[band][v] > 0 else targets_median_4[band]
                metadata[i]['stats'][band]['src_median_5'] = medians_5[band][v] if medians_5[band][v] > 0 else targets_median_5[band]
                metadata[i]['stats'][band]['src_median_6'] = medians_6[band][v] if medians_6[band][v] > 0 else targets_median_6[band]

                metadata[i]['stats'][band]['src_madstd'] = madstds[band][v] if madstds[band][v] > 0 else targets_madstd[band]
                metadata[i]['stats'][band]['src_madstd_4'] = madstds_4[band][v] if madstds_4[band][v] > 0 else targets_madstd_4[band]
                metadata[i]['stats'][band]['src_madstd_5'] = madstds_5[band][v] if madstds_5[band][v] > 0 else targets_madstd_5[band]
                metadata[i]['stats'][band]['src_madstd_6'] = madstds_6[band][v] if madstds_6[band][v] > 0 else targets_madstd_6[band]

                metadata[i]['stats'][band]['target_median'] = targets_median[band]
                metadata[i]['stats'][band]['target_median_4'] = targets_median_4[band]
                metadata[i]['stats'][band]['target_median_5'] = targets_median_5[band]
                metadata[i]['stats'][band]['target_median_6'] = targets_median_6[band]

                metadata[i]['stats'][band]['target_madstd'] = targets_madstd[band]
                metadata[i]['stats'][band]['target_madstd_4'] = targets_madstd_4[band]
                metadata[i]['stats'][band]['target_madstd_5'] = targets_madstd_5[band]
                metadata[i]['stats'][band]['target_madstd_6'] = targets_madstd_6[band]
        
    # Clear memory of scl images
    for j in range(len(metadata)):
        metadata[j]['scl'] = None

    if output_tracking is True:
        array_to_raster(tracking_array.astype('uint8'), reference_raster=best_image['path']['20m']['B04'], out_raster=os.path.join(out_dir, f"tracking_{out_name}.tif"), dst_projection=dst_projection)

    if output_scl is True:
        array_to_raster(master_scl.astype('uint8'), reference_raster=best_image['path']['20m']['B04'], out_raster=os.path.join(out_dir, f"scl_{out_name}.tif"), dst_projection=dst_projection)
        
    if output_quality is True:
        array_to_raster(master_quality.astype('float32'), reference_raster=best_image['path']['20m']['B04'], out_raster=os.path.join(out_dir, f"quality_{out_name}.tif"), dst_projection=dst_projection)

    # Resample scl and tracking array
    tracking_array = raster_to_array(resample(array_to_raster(tracking_array, reference_raster=best_image['path']['20m']['B04']), reference_raster=best_image['path']['10m']['B04']))
    master_scl = raster_to_array(resample(array_to_raster(master_scl, reference_raster=best_image['path']['20m']['B04']), reference_raster=best_image['path']['10m']['B04']))

    # Run a mode filter on the tracking array
    if filter_tracking is True and multiple_images is True:
        if verbose: print('Filtering tracking array..')

        tracking_array = mode_filter(tracking_array, 7).astype('uint8')

    # Feather the edges between joined images (ensure enough valid pixels are on each side..)
    if feather is True and multiple_images is True:
        feathers = {}
        
        print('Precalculating classification feathers..')
        feather_rest = feather_s2_filter(master_scl, np.array([0, 1, 2, 3, 7, 8, 9, 10, 11], dtype='intc'), feather_scl).astype('float32')
        feather_4 = feather_s2_filter(master_scl, np.array([4], dtype='intc'), feather_scl).astype('float32')
        feather_5 = feather_s2_filter(master_scl, np.array([5], dtype='intc'), feather_scl).astype('float32')
        feather_6 = feather_s2_filter(master_scl, np.array([6], dtype='intc'), feather_scl).astype('float32')

        if verbose: print('Precalculating inter-layer feathers..')
        for i in processed_images_indices:
            feathers[str(i)] = feather_s2_filter(tracking_array, np.array([i], dtype='intc'), feather_dist).astype('float32')

    if match_mean is True and feather is False and len(processed_images_indices) > 1:
        mask_4 = (master_scl == 4)
        mask_5 = (master_scl == 5)
        mask_6 = (master_scl == 6)
        mask_rest = (master_scl != 4) & (master_scl != 5) & (master_scl != 6)

    bands_to_output = ['B02', 'B03', 'B04', 'B08']
    if verbose: print('Merging band data..')
    for band in bands_to_output:
        if verbose: print(f'Writing: {band}..')
        base_image = raster_to_array(metadata[0]['path']['10m'][band]).astype('float32')

        for i in processed_images_indices:

            if match_mean and len(processed_images_indices) > 1:
                src_med = metadata[i]['stats'][band]['src_median']
                src_med_4 = metadata[i]['stats'][band]['src_median_4']
                src_med_5 = metadata[i]['stats'][band]['src_median_5']
                src_med_6 = metadata[i]['stats'][band]['src_median_6']

                src_mad = metadata[i]['stats'][band]['src_madstd']
                src_mad_4 = metadata[i]['stats'][band]['src_madstd_4']
                src_mad_5 = metadata[i]['stats'][band]['src_madstd_5']
                src_mad_6 = metadata[i]['stats'][band]['src_madstd_6']

                target_med = metadata[i]['stats'][band]['target_median']
                target_med_4 = metadata[i]['stats'][band]['target_median_4']
                target_med_5 = metadata[i]['stats'][band]['target_median_5']
                target_med_6 = metadata[i]['stats'][band]['target_median_6']

                target_mad = metadata[i]['stats'][band]['target_madstd']
                target_mad_4 = metadata[i]['stats'][band]['target_madstd_4']
                target_mad_5 = metadata[i]['stats'][band]['target_madstd_5']
                target_mad_6 = metadata[i]['stats'][band]['target_madstd_6']

            if i == 0:
                if match_mean and len(processed_images_indices) > 1:
                    dif = base_image - src_med
                    dif_4 = base_image - src_med_4
                    dif_5 = base_image - src_med_5
                    dif_6 = base_image - src_med_6

                    if feather is True and len(processed_images_indices) > 1:
                        base_image = (((dif * target_mad) / src_mad) + target_med) * feather_rest
                        base_image = np.add(base_image, (((dif_4 * target_mad_4) / src_mad_4) + target_med_4) * feather_4)
                        base_image = np.add(base_image, (((dif_5 * target_mad_5) / src_mad_5) + target_med_5) * feather_5)
                        base_image = np.add(base_image, (((dif_6 * target_mad_6) / src_mad_6) + target_med_6) * feather_6)

                    else:
                        base_image_rest = ((dif * target_mad) / src_mad) + target_med
                        base_image_4 = ((dif_4 * target_mad_4) / src_mad_4) + target_med_4
                        base_image_5 = ((dif_5 * target_mad_5) / src_mad_5) + target_med_5
                        base_image_6 = ((dif_6 * target_mad_6) / src_mad_6) + target_med_6

                        base_image = np.where(mask_rest, base_image_rest, base_image)
                        base_image = np.where(mask_4, base_image_4, base_image)
                        base_image = np.where(mask_5, base_image_5, base_image)
                        base_image = np.where(mask_6, base_image_6, base_image)
                
                    base_image = np.where(base_image >= 0, base_image, 0)

                if feather is True and len(processed_images_indices) > 1:
                    base_image = base_image * feathers[str(i)]

            else:
                add_band = raster_to_array(metadata[i]['path']['10m'][band]).astype('float32')
                
                if match_mean:       
                    dif = add_band - src_med
                    dif_4 = add_band - src_med_4
                    dif_5 = add_band - src_med_5
                    dif_6 = add_band - src_med_6
                    
                    if feather is True:
                        add_band = (((dif * target_mad) / src_mad) + target_med) * feather_rest
                        add_band = np.add(add_band, (((dif_4 * target_mad_4) / src_mad_4) + target_med_4) * feather_4)
                        add_band = np.add(add_band, (((dif_5 * target_mad_5) / src_mad_5) + target_med_5) * feather_5)
                        add_band = np.add(add_band, (((dif_6 * target_mad_6) / src_mad_6) + target_med_6) * feather_6)
                    else:             
                        add_band_rest = ((dif * target_mad) / src_mad) + target_med
                        add_band_4 = ((dif_4 * target_mad_4) / src_mad_4) + target_med_4
                        add_band_5 = ((dif_5 * target_mad_5) / src_mad_5) + target_med_5
                        add_band_6 = ((dif_6 * target_mad_6) / src_mad_6) + target_med_6
                        
                        add_band = np.where(mask_rest, add_band_rest, add_band)
                        add_band = np.where(mask_4, add_band_4, add_band)
                        add_band = np.where(mask_5, add_band_5, add_band)
                        add_band = np.where(mask_6, add_band_6, add_band)
                    
                    add_band = np.where(add_band >= 0, add_band, 0)

                if feather is True:
                    base_image = np.add(base_image, (add_band * feathers[str(i)]))
                else:
                    base_image = np.where(tracking_array == i, add_band, base_image).astype('float32')

        array_to_raster(np.rint(base_image).astype('uint16'), reference_raster=best_image['path']['10m'][band], out_raster=os.path.join(out_dir, f"{band}_{out_name}.tif"), dst_projection=dst_projection)

    if verbose: print(f'Completed mosaic in: {round((time() - start_time) / 60, 1)}m')
