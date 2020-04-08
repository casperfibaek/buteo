import sys; sys.path.append('..'); sys.path.append('../lib/')
import numpy as np
import cv2
import os
from math import pow, sqrt
from glob import glob
from shutil import copyfile
from lib.raster_io import raster_to_metadata, raster_to_array, clip_raster, array_to_raster
from lib.vector_io import intersection_rasters, vector_mask
from lib.stats_filters import mean_filter
from lib.utils_core import madstd


in_dir = '/mnt/c/Users/caspe/Desktop/tests/'
out_dir = '/mnt/c/Users/caspe/Desktop/tests/processed/'
images = glob(in_dir + '*.tif')

for i in range(len(images)):
    image = images[i]
    out_name = os.path.basename(os.path.normpath(image))
    count = 0
    sections = []
    med_ratios = []
    mad_ratios = []
    weights = []
    img_med = []
    img_mad = []
    
    distance_images = []
    
    for compare_image in images:
        if image == compare_image:
            continue

        intersection = intersection_rasters(image, compare_image)
        
        if intersection is False:
            continue

        comp_name = os.path.basename(os.path.normpath(compare_image))
    
        base_masked = raster_to_array(vector_mask(intersection, image))
        base_masked_distance = cv2.distanceTransform(base_masked, cv2.DIST_L2, 5)
        max_dist = sqrt(pow(base_masked.shape[0], 2) + pow(base_masked.shape[1], 2))
        
        distance_images.append(1 - (base_masked_distance / max_dist))
        
        base_clipped = clip_raster(image, align=True, cutline=intersection, cutline_all_touch=True)
        base_image = raster_to_array(base_clipped)
        comp_clipped = clip_raster(compare_image, align=True, cutline=intersection, cutline_all_touch=True)
        comp_image = raster_to_array(comp_clipped)
        
        if base_image.shape != comp_image.shape:
            rows = base_image.shape[0] if base_image.shape[0] < comp_image.shape[0] else comp_image.shape[0]
            cols = base_image.shape[1] if base_image.shape[1] < comp_image.shape[1] else comp_image.shape[1]
            
            base_image = base_image[1:rows-1, 1:cols-1]
            comp_image = comp_image[1:rows-1, 1:cols-1]

        weights.append(comp_image.size)

        images_mask = (base_image == 0) | (comp_image == 0)
        
        base_image = np.ma.array(base_image, mask=images_mask)
        base_med, base_mad = madstd(base_image)
        # base_med = base_image.mean()
        # base_mad = base_image.std()

        img_med.append(base_med)
        img_mad.append(base_mad)
        
        comp_image = np.ma.array(comp_image, mask=images_mask)
        comp_med, comp_mad = madstd(comp_image)
        # comp_med = comp_image.mean()
        # comp_mad = comp_image.std()
        
        sections.append({
            'base_med': base_med,
            'base_mad': base_mad,
            'comp_med': comp_med,
            'comp_mad': comp_mad,
        })
        
        count  += 1

    summed = np.zeros(distance_images[0].shape, dtype='float32')
    for img in distance_images:
        summed = summed + img

    target_med = np.zeros(distance_images[0].shape, dtype='float32')
    target_mad = np.zeros(distance_images[0].shape, dtype='float32')

    for s, p in enumerate(sections):
        med_ratios.append(sections[s]['comp_med'] / sections[s]['base_med'])
        mad_ratios.append(sections[s]['comp_mad'] / sections[s]['base_mad'])

        target_med = target_med + ((sections[s]['comp_med'] / sections[s]['base_med']) * sections[s]['base_med']) * (distance_images[s] / summed)
        target_mad = target_mad + ((sections[s]['comp_mad'] / sections[s]['base_mad']) * sections[s]['base_mad']) * (distance_images[s] / summed)

    # array_to_raster(med_array, reference_raster=image, out_raster=out_dir + f"med_{out_name}_{comp_name}.tif")
    # array_to_raster(mad_array, reference_raster=image, out_raster=out_dir + f"mad_{out_name}_{comp_name}.tif")

    # exit()
    
    if count == 0:
        print('No overlaps found, copying input..')
        copyfile(image, out_dir + out_name)
    else:
        # weights_normalized = []
        # for weight in weights:
        #     weights_normalized.append(weight / sum(weights))

        # src_med = 0
        # src_mad = 0
        # target_med_ratio = 0
        # target_mad_ratio = 0
        # for j in range(len(med_ratios)):
            # src_med += (weights_normalized[j] * img_med[j])
            # src_mad += (weights_normalized[j] * img_mad[j])
            # target_med_ratio += (weights_normalized[j] * med_ratios[j])
            # target_mad_ratio += (weights_normalized[j] * mad_ratios[j])
        
        array = raster_to_array(image)
        array = np.ma.array(array, mask=array==0)

        src_med, src_mad = madstd(array)
        # src_med = array.mean()
        # src_mad = array.std()

        # target_med = src_med * target_med_ratio
        # target_mad = src_mad * target_mad_ratio
        
        dif = np.ma.subtract(array, src_med)
        out_image = ((dif * target_mad) / src_mad) + target_med

        array_to_raster(np.rint(out_image).astype('uint16'), reference_raster=image, out_raster=out_dir + out_name)
