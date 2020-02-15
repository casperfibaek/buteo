#!/usr/bin/env python

import datetime
import glob
import multiprocessing
import numpy as np
import os
from scipy import ndimage
from scipy import interpolate
import subprocess

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

import sen2mosaic.IO

import pdb


# global scenes_tile

################################################################
### Functions for Sentinel-2 data compositing and mosaicking ###
################################################################


##########################
### Internal functions ###
##########################

def _nan_percentile(arr, quant):
    """
    Function to calculate a percentile along the first axis of a 3d array, much faster than np.nanpercentile.
    Modified with permission from: https://krstn.eu/np.nanpercentile()-there-has-to-be-a-faster-way/
    
    Args:
        arr: Three dimensional numpy array (first dimension to be reduced)
        quant: Percentile, in percent
    Returns:
        A two dimensional numpy array containing data from percentile quant
    """
    
    assert quant >= 0 and quant <= 100, "Quantile must be between 0 % and 100 %."
    
    def _zvalue_from_index(arr, ind):
        """
        Private helper function to work around the limitation of np.choose() by employing np.take()
        arr has to be a 3D array
        ind has to be a 2D array containing values for z-indicies to take from arr
        See: http://stackoverflow.com/a/32091712/4169585
        This is faster and more memory efficient than using the ogrid based solution with fancy indexing.
        """
        # get number of columns and rows
        _,nC,nR = arr.shape
        
        # Get linear indices and extract elements with np.take()
        #idx = nC*nR*ind + nC*np.arange(min(nC,nR))[:,None] + np.arange(max(nC,nR))
        idx = nC*nR*ind + np.arange(nC*nR).reshape((nC,nR))
        
        return np.take(arr, idx)
    
    # Valid (non NaN) observations along the first axis
    valid_obs = np.sum(np.isfinite(arr), axis=0)
    
    # Replace NaN with maximum
    max_val = np.nanmax(arr)
    arr[np.isnan(arr)] = max_val
    
    # Sort - former NaNs will move to the end
    arr = np.sort(arr, axis=0)
    
    # Build output array
    quant_arr = np.zeros(shape=(arr.shape[1], arr.shape[2]))

    # Desired position as well as floor and ceiling of it
    k_arr = (valid_obs - 1) * (quant / 100.0)
    f_arr = np.floor(k_arr).astype(np.int32)
    c_arr = np.ceil(k_arr).astype(np.int32)
    
    # Floor == Ceil
    fc_equal_k_mask = f_arr == c_arr

    # Linear interpolation (like numpy percentile) takes the fractional part of desired position
    floor_val = _zvalue_from_index(arr = arr, ind = f_arr) * (c_arr - k_arr)
    ceil_val = _zvalue_from_index(arr = arr, ind = c_arr) * (k_arr - f_arr)
    
    quant_arr = floor_val + ceil_val
    quant_arr[fc_equal_k_mask] = _zvalue_from_index(arr = arr, ind = k_arr.astype(np.int32))[fc_equal_k_mask]  # if floor == ceiling take floor value

    return np.round(quant_arr,0).astype(np.uint16)


def _makeBlocks(band, scene, step = 2000, percentile = 25., improve_mask = False, masked_vals = [], temp_dir = '/tmp'):
    '''
    Function to build a series of blocks of size (step, step) to enable multiprocessing and prevent overloading memory
    
    Args:
        band: Image band (e.g. 'B02')
        scene: An example Sentinel-2 LoadScene() object
        step: Step size, an integer determining block size
        percentile: Percentile of reflectance to take from valid pixels
        improve_mask: Apply improvements to Sentinel-2 cloud mask. Not generally recommended.
    
    Returns:
        A list of blocks to be processed by _doComposite()
    '''
    
    blocks = []
    
    for col in range(0, scene.metadata.ncols, step):
        col_step = step if col + step <= scene.metadata.ncols else scene.metadata.ncols - col
        for row in range(0, scene.metadata.nrows, step):
            row_step = step if row + step <= scene.metadata.nrows else scene.metadata.nrows - row
            if row_step ==0 or col_step ==0: pdb.set_trace()
            blocks.append([band, col, col_step, row, row_step, percentile, improve_mask, masked_vals, temp_dir])
     
    return blocks


def _doComposite(input_list):
    '''
    Function to build a cloud-free composite image for a block based on a percentile of surface reflectance.
    Internal function for buildComposite.
    
    Args:
        input_list: A list containing band, col, col_step, row, row_step from _makeBlocks(), percentile, improve_mask, and masked_vals 
    Returns:
        A composite image for input block
    '''
    
    band, col, col_step, row, row_step, percentile, improve_mask, masked_vals, temp_dir = input_list
    
    # Mask stack
    m = np.zeros((len(scenes_tile), col_step, row_step), dtype = np.uint8)
    b = np.zeros((len(scenes_tile), col_step, row_step), dtype = np.float32)
    
    for n, scene in enumerate(scenes_tile):
        
        m[n,:,:] = scene.getMask(improve = improve_mask, chunk = [row,col,row_step,col_step], temp_dir = temp_dir)
        
        if m[n,:,:].sum() == 0: continue
        
        b[n,:,:] = scene.getBand(band, chunk = [row,col,row_step,col_step])
    
    # If nodata in the entire chunk, skip processing
    if m.sum() == 0: return np.zeros_like(b[0,:,:]).astype(np.uint16), np.zeros_like(b[0,:,:]).astype(np.uint8)
    
    bm = np.ma.array(b, mask = np.ones_like(m,dtype=np.bool))
    
    # Build output arrays
    nodata = np.ones_like(b[0,:,:], dtype = np.bool)
    slc = np.zeros_like(b[0,:,:], dtype = np.uint8)
    slc_count = np.zeros_like(b[0,:,:], dtype = np.uint8)
    slc_assigned = np.zeros_like(b[0,:,:], dtype = np.bool)
    
    # Add pixels in order of desirability
    for n, vals in enumerate([[4,5,6], [2,7,11], [1,3,8,10], [9]]):
        
        # Strip masked values from vals
        for masked_val in masked_vals:
            if masked_val in vals:
                vals.remove(masked_val)
        
        # Skip if all vals are in masked_vals
        if len(vals) == 0: continue
        
        bm[:,nodata] = np.ma.array(b, mask = np.isin(m, vals) == False)[:,nodata]
        
        # Calculate modal SLC value
        for val in vals:
            this_count = (m == val).sum(axis = 0)
            slc[np.logical_and(this_count > slc_count, slc_assigned == False)] = val
            slc_count[this_count > slc_count] = this_count[this_count > slc_count]
        
        slc_assigned[slc_count != 0] = True
        
        nodata = (bm.mask == False).sum(axis = 0) == 0
            
    bm = np.ma.filled(bm, np.nan)
    
    bm = _nan_percentile(bm, percentile)
    
    # Set nodata to 0
    bm[nodata] = 0
    
    return bm, slc


###########################################
### Functions to improve mosaic quality ###
###########################################

def _histogramMatch(source, reference):
    """       
    Adjust the values of a source array so that its histogram matches that of a reference array
    
    Modified from: https://github.com/mapbox/rio-hist/blob/master/rio_hist/match.py
    
    Args:
        source: A numpy array of Sentinel-2 data
        reference: A numpy array of Sentinel-2 data to match colours to

    Returns:
        target: A numpy array array with the same shape as source
    """
        
    orig_shape = source.shape
    source = source.ravel()

    if np.ma.is_masked(reference):
        reference = reference.compressed()
    else:
        reference = reference.ravel()

    # Get the set of unique pixel values
    s_values, s_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    
    # and those to match to
    r_values, r_counts = np.unique(reference, return_counts=True)
    s_size = source.size

    if np.ma.is_masked(source):
        mask_index = np.ma.where(s_values.mask)
        s_size = np.ma.where(s_idx != mask_index[0])[0].size
        s_values = s_values.compressed()
        s_counts = np.delete(s_counts, mask_index)

    # Calculate cumulative distribution
    s_quantiles = np.cumsum(s_counts).astype(np.float64) / s_size
    r_quantiles = np.cumsum(r_counts).astype(np.float64) / reference.size

    # Find values in the reference corresponding to the quantiles in the source
    interp_r_values = np.interp(s_quantiles, r_quantiles, r_values)

    if np.ma.is_masked(source):
        interp_r_values = np.insert(interp_r_values, mask_index[0], source.fill_value)

    # using the inverted source indicies, pull out the interpolated pixel values
    target = interp_r_values[s_idx]

    if np.ma.is_masked(source):
        target = np.ma.masked_where(s_idx == mask_index[0], target)
        target.fill_value = source.fill_value

    return target.reshape(orig_shape)


def _colourBalance(image, reference, aggressive = True, verbose = False):
    '''
    Perform colour balancing between a new and reference image.
    '''
    
    # Calculate overlap with other images
    overlap = np.logical_and(image.mask == False, reference.mask == False)
    
    # Calculate percent overlap between images
    this_overlap = float(overlap.sum()) / (image.mask == False).sum()
        
    if this_overlap > 0.02 and this_overlap <= 0.5 and aggressive:
        
        if verbose: print('        colour scaling')
                
        # Gain compensation (simple inter-scene correction)                    
        this_intensity = np.mean(image[overlap])
        ref_intensity = np.mean(reference[overlap])
        
        image[image.mask == False] = np.round(image[image.mask == False] * (ref_intensity/this_intensity),0).astype(np.uint16)
        
    elif this_overlap > 0.5:
        
        if verbose: print('        colour matching')
        
        image = _histogramMatch(image, reference)
        
    else:
        
        if verbose: print('        colour adding')
    
    return image


#########################
### Primary functions ###
#########################

def buildComposite(source_files, band, md_dest, resolution = 20, level = '2A', output_dir = os.getcwd(), output_name = 'mosaic', start = '20150101', end = datetime.datetime.today().strftime('%Y%m%d'), step = 2000, improve_mask = False, processes = 1, percentile = 25., colour_balance = False, masked_vals = 'auto', output_mask = True, temp_dir = '/tmp', verbose = False, resampling = 0):
    """
    
    Function to generate seamless mosaics from a list of Sentinel-2 level-1C/2A input files.
        
    Args:
        source_files: A list of level 1C/2A input files.
        extent_dest: List desciribing corner coordinate points in destination CRS [xmin, ymin, xmax, ymax].
        EPSG_dest: EPSG code of destination coordinate reference system. Must be a UTM projection. See: https://www.epsg-registry.org/ for codes.
        level: Sentinel-2 level 1C '1C' or level 2A '2A' input data.
        start: Start date to process, in format 'YYYYMMDD' Defaults to start of Sentinel-2 era.
        end: End date to process, in format 'YYYYMMDD' Defaults to today's date.
        resolution: Resolution band 10, 20, or 60 m band to use. Defaults to 20.
        improve_mask: Set True to apply improvements Sentinel-2 cloud mask. Not generally recommended.
        processes: Number of processes to run similtaneously. Defaults to 1.
        output_dir: Optionally specify an output directory.
        output_name: Optionally specify a string to precede output file names. Defaults to 'mosaic'.
        masked_vals: List of SLC mask values to not include in the final mosaic. Defaults to 'auto', which masks everything except [4,5,6]
        temp_dir: Directory to temporarily write L1C mask files. Defaults to /tmp
        verbose: Make script verbose (set True).
    """
    
    # Test input formatting
    assert len(source_files) >= 1, "No level %s source files in specified location."%str(level)
    assert resolution in [0, 10, 20, 60], "Resolution must be 10, 20, or 60 m."
    assert type(improve_mask) == bool, "improve_mask can only be set to True or False."
    assert level in ['1C', '2A'], "Sentinel-2 processing level must be either '1C' or '2A'."
    assert percentile >=0 and percentile <=100, "Percentile cannot be set less than 0% or greater than 100%."
            
    # Set values to be masked
    if masked_vals == 'auto': masked_vals = [0,9]#[0,1,2,3,7,8,9,10,11]
    if masked_vals == 'none': masked_vals = []
    assert type(masked_vals) == list, "Masked values must be a list of integers, or set to 'auto' or 'none'."
    
    # Test that output directory is writeable
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    assert os.path.exists(output_dir), "Output directory (%s) does not exist."%output_dir
    assert os.access(output_dir, os.W_OK), "Output directory (%s) does not have write permission. Try setting a different output directory, or changing permissions with chmod."%output_dir
    
    for m in masked_vals:
        assert type(m) == int, "Masked values must all be integers."

    # Load all Sentinel-2 input datasets
    scenes = sen2mosaic.IO.loadSceneList(source_files, resolution = resolution, md_dest = md_dest, start = start, end = end, level = level, sort_by = 'date')
    
    # It's only worth processing a tile if at least one input image is inside tile
    if len(scenes) == 0:
        raise IOError("No data inside specified output area or date range for resolution %s. Make sure you specified your bouding box in the correct order (i.e. xmin ymin xmax ymax), EPSG code correctly, and that start and end dates are in the format YYYYMMDD. Continuing."%str(resolution))
        
    # Print reassuring statement
    if verbose: print("Found %s scenes matching output criteria."%str(len(scenes)))
        
    # Build blank output images    
    composite_out = md_dest.createBlankArray(dtype = np.uint16)
    slc_out = md_dest.createBlankArray(dtype = np.uint8)
        
    # Make list of scenes accessible
    global scenes_tile
    
    # Process one Sentinel-2 tile at a time
    for tile in np.unique([s.tile for s in scenes]):
        
        scenes_tile = np.array(scenes)[np.array([s.tile for s in scenes]) == tile]
        
        scene = scenes_tile[0]
                
        composite = np.zeros((scene.metadata.ncols, scene.metadata.nrows), dtype = np.uint16)
        slc = np.zeros((scene.metadata.ncols, scene.metadata.nrows), dtype = np.uint8)
        
        blocks = _makeBlocks(band, scene, step = step, percentile = percentile, improve_mask = improve_mask, masked_vals = masked_vals, temp_dir = temp_dir)        
        
        # Do the compositing
        if processes == 1:
            composite_parts = [_doComposite(block) for block in blocks]
        else:
            pool = multiprocessing.Pool(processes)
            composite_parts = pool.map(_doComposite, blocks)
            pool.close()
        
        
        # Reconsitute image
        for n, block in enumerate(blocks):
            band, col, col_step, row, row_step, _, _, _, _ = block
            composite[col:col+col_step,row:row+row_step] = composite_parts[n][0]
            slc[col:col+col_step,row:row+row_step] = composite_parts[n][1]
        
        # Reproject to match output array
        composite_rep = sen2mosaic.IO.reprojectBand(scene, composite, md_dest, dtype = 3, resampling = resampling)
        slc_rep = sen2mosaic.IO.reprojectBand(scene, slc, md_dest, dtype = 1, resampling = 0)
        
        # Do optional colour balancing
        if colour_balance:
            composite_rep = _colourBalance(np.ma.array(composite_rep, mask = composite_rep == 0), np.ma.array(composite_out, mask = composite_out == 0), verbose = verbose)
            composite_rep = composite_rep.filled(0)
        
        # Add pixels to the output mosaic
        sel = composite_rep!=0
        composite_out[sel] = composite_rep[sel]
        slc_out[sel] = slc_rep[sel]
    
    # Output composite image
    sen2mosaic.IO.createGdalDataset(md_dest, data_out = composite_out, filename = '%s/%s_R%sm_%s.tif'%(output_dir, output_name, str(scene.resolution), band), driver='GTiff', nodata = 0, options = ['COMPRESS=LZW'])        
    
    # Output mask
    if output_mask:
        sen2mosaic.IO.createGdalDataset(md_dest, data_out = slc_out, filename = '%s/%s_R%sm_SLC.tif'%(output_dir, output_name, str(scene.resolution)), driver='GTiff', options = ['COMPRESS=LZW'])        
    
    return composite_out, slc_out
        

def buildVRT(red_band, green_band, blue_band, output_path):
    """
    Builds a three band RGB vrt for image visualisation. Outputs a .VRT file.
    
    Args:
        red_band: Filename to add to red band
        green_band: Filename to add to green band
        blue_band: Filename to add to blue band
        output_name: Path to output file
    """
    
    # Remove trailing / from output directory name if present
    output_path = output_path.rstrip('/')
    
    # Ensure output name is a VRT
    if output_path[-4:] != '.vrt':
        output_path += '.vrt'
    
    command = ['gdalbuildvrt', '-separate', '-overwrite']
    command += [output_path, red_band, green_band, blue_band]
    
    subprocess.call(command)


if __name__ == '__main__':
    '''
    '''
        
    print('The sen2mosaic command line interface has been moved! Please use scripts in .../sen2mosaic/cli/ to operate sen2mosaic from the command line.')
    
