#!/usr/bin/env python

import datetime
import numpy as np
import os
import subprocess

import pdb

import sen1mosaic.IO
import sen2mosaic.IO

#########################
### Private functions ###
#########################

def _createOutputArray(md, dtype = np.uint16):
    '''
    Create an output array from metadata dictionary.
    
    Args:
        md: A metadata object from utiltities.Metadata()
    
    Returns:
        A numpy array sized to match the specification of the metadata dictionary.
    '''
    
    output_array = np.zeros((md.nrows, md.ncols), dtype = dtype)
    
    return output_array


def _updateDataArray(data_out, data_resampled, mask = None, action = 'sum'):
    '''
    Function to update contents of output array based on image_n array.
    
    Args:
        data_out: A numpy array representing the band data to be output.
        data_resampled: A numpy array containing resampled band data to be added to data_out.
        action: 'sum', 'min', or 'max', which respectively adds data_resampled to data_out, replaces pixels in data_out with data_resampled where data_resampled < data_out, and replaces pixels in data_out with data_resampled where data_resampled > data_out.
    
    Returns:
        The data_out array with pixels from data_resampled added.
        
    '''
    
    if mask is None: mask = data_resampled == 0
    
    assert action in ['sum', 'min', 'max'], "Variable 'action' must be set to 'sum', 'min' or 'max'. It was set to %s."%str(action)
    
    # Add usable data to data_out array   
    if action == 'sum':
        #mask = data_resampled != 0
        data_out[mask == False] += data_resampled[mask == False]
    elif action == 'max':
        use = np.logical_and(np.logical_or(data_resampled > data_out, mask == False), data_resampled != 0)
        data_out[use] = data_resampled[use]
    elif action == 'min':
        use = np.logical_or(data_resampled < data_out, mask == False)
        data_out[use] = data_resampled[use]

    return data_out


def _generateDataArray(scenes, pol, md_dest, output_dir = os.getcwd(), output_name = 'S1_output', verbose = False):
    """generateDataArray(source_files, pol, md_dest, output_dir = os.getcwd(), output_name = 'S1_output', verbose = False)
    
    Function which generates an output GeoTiff file from list of pre-processed S1 source files for a specified output polarisation and extent.

    Args:
        scenes: A list of pre-processed S1 input files.
        pol: Polarisation to process ('VV' or 'VH')
        md_dest: Dictionary from buildMetaDataDictionary() containing output projection details.
        output_dir: Directory to write output files. Defaults to current working directory.
        output_name: Optionally specify a string to prepend to output files. Defaults to 'S1_output'.
        
    Returns:
        A string with the filename pattern. Returns 'NODATA' where not valid input images.
    """
    
    # Create array to contain output array for this band. Together these arrays are used to calculate the mean, min, max and standard deviation of input images.
    data_num = _createOutputArray(md_dest, dtype = np.int16) # To track number of images for calculating mean
    data_sum = _createOutputArray(md_dest, dtype = np.float32) # To track sum of input images
    data_var = _createOutputArray(md_dest, dtype = np.float32) # To track sum of variance of input images
    data_min = _createOutputArray(md_dest, dtype = np.float32) # To track sum of max value of input images
    data_max = _createOutputArray(md_dest, dtype = np.float32) # To track sum of min value of input images
    
    data_date = _createOutputArray(md_dest, dtype = np.float32) # Data from each image
                   
    # For each source file
    for n, scene in enumerate(scenes):
        
        if verbose: print('    Adding pixels from %s'%scene.filename.split('/')[-1])
        
        
        # Update output arrays if we're finished with the previous date. Skip on first iteration as there's no data yet.
        if n != 0:
            if scene.datetime.date() != last_date:
                data_num = _updateDataArray(data_num, (data_date != 0.) * 1, action = 'sum')
                data_sum = _updateDataArray(data_sum, data_date, action = 'sum')
                data_var = _updateDataArray(data_var, data_date ** 2, action = 'sum')
                data_min = _updateDataArray(data_min, data_date, action = 'min')
                data_max = _updateDataArray(data_max, data_date, action = 'max')
                
        # Update date for next loop
        last_date = scene.datetime.date()
        
        # Load data
        data_resampled = scene.getBand(pol, md = md_dest)
        mask = scene.getMask(pol, md = md_dest) 
                
        # Update array for this date (allowing only 1 measurement per date to be included in sum)
        data_date = _updateDataArray(data_date, data_resampled, action = 'min')
        
        # Tidy up
        ds_source = None
        ds_dest = None
   
    # Update output arrays on final loop
    data_num = _updateDataArray(data_num, (data_date != 0.) * 1, action = 'sum')
    data_sum = _updateDataArray(data_sum, data_date, action = 'sum')
    data_var = _updateDataArray(data_var, data_date ** 2, action = 'sum')
    data_min = _updateDataArray(data_min, data_date, action = 'min')
    data_max = _updateDataArray(data_max, data_date, action = 'max')
    
    # Get rid of zeros in cases of no data
    data_num[data_num==0] = 1
    
    # Calculate mean of input data
    data_mean = data_sum / data_num.astype(np.float32)
    
    # Calculate std of input data (See: https://stackoverflow.com/questions/5543651/computing-standard-deviation-in-a-stream). Where standard deviation undefined (< 2 samples), set to 0.
    data_std = np.zeros_like(data_mean)
    data_std[data_num > 1] = ((data_num * data_var - np.abs(data_sum) * data_sum)[data_num > 1]) / ((np.abs(data_sum) * (data_num - 1))[data_num > 1])
    data_std = np.sqrt(data_std)
    data_std[data_num < 2] = 0.
    
    if verbose: print('Outputting polarisation %s'%pol)
    
    # Generate default output filename
    filename = '%s/%s_%s_%s_R%sm.tif'%(output_dir, output_name, '%s', pol, str(md_dest.res))
    
    # Output files (mean, stdev, max, min)
    ds_out = sen2mosaic.IO.createGdalDataset(md_dest, data_out = data_mean, filename = filename%'mean', driver='GTiff', dtype = 6, options = ['COMPRESS=LZW'])
    ds_out = sen2mosaic.IO.createGdalDataset(md_dest, data_out = data_std, filename = filename%'stdev', driver='GTiff', dtype = 6, options = ['COMPRESS=LZW'])    
    ds_out = sen2mosaic.IO.createGdalDataset(md_dest, data_out = data_max, filename = filename%'max', driver='GTiff', dtype = 6, options = ['COMPRESS=LZW'])
    ds_out = sen2mosaic.IO.createGdalDataset(md_dest, data_out = data_min, filename = filename%'min', driver='GTiff', dtype = 6, options = ['COMPRESS=LZW'])

    return filename


############################################
### Functions to build a composite image ###
############################################

def buildComposite(source_files, pol, md_dest, start = '20140101', end = datetime.datetime.today().strftime('%Y%m%d'), output_name = 'S1_output', output_dir = os.getcwd(), verbose = False):
    '''
    '''
    
    if verbose: print('Doing polarisation %s'%pol)
    
    # Load metadata for all Sentinel-1 datasets    
    scenes = sen1mosaic.IO.loadSceneList(source_files, md_dest = md_dest, pol = pol, start = start, end = end, sort = True)
    
    # It's only worth processing a tile if at least one input image is inside tile
    if len(scenes) == 0:
        raise IOError("No data inside specified tile for polarisation %s."%pol)
        
    # Combine pixels into output images for each band
    filename = _generateDataArray(scenes, pol, md_dest, output_dir = output_dir, output_name = output_name, verbose = verbose)
    
    return filename


def buildVVVH(VV_file, VH_file, md, output_dir = os.getcwd(), output_name = 'S1_output'):
    """buildVVVH(VV_file, VH_file, md, output_dir = os.getcwd(), output_name = 'S1_output')
    
    Function to build a VV/VH array and output to GeoTiff.
    
    Args:
        VV_file: Path to a VV output Geotiff
        VH_file: Path to a VH output Geotiff
        output_dir: Directory to write output files. Defaults to current working directory.
        output_name: Optionally specify a string to prepend to output files. Defaults to 'S1_output'.
    
    Returns:
        Path to VV/VH GeoTiff
    """
    
    from osgeo import gdal
    
    # Load datasets
    ds_VV = gdal.Open(VV_file,0)
    ds_VH = gdal.Open(VH_file,0)
    
    data_VV = ds_VV.ReadAsArray()
    data_VH = ds_VH.ReadAsArray()
    
    mask = np.logical_or(data_VV == 0., data_VH == 0.)
    
    data_VV[data_VV >= 0] = -0.00001
    data_VH[data_VH >= 0] = -0.00001

    VV_VH = data_VV / data_VH
    
    VV_VH[mask] = 0.
        
    # Output to GeoTiff
    res = str(int(round(ds_VV.GetGeoTransform()[1])))
    filename = '%s/%s_%s_VVVH_R%sm.tif'%(output_dir, output_name, 'mean', res)
    
    ds_out = sen2mosaic.IO.createGdalDataset(md, data_out = VV_VH, filename = filename, driver='GTiff', dtype = 6, options = ['COMPRESS=LZW'])
            
    return filename

    
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
        
    print('The sen1mosaic command line interface has been moved! Please use scripts in .../sen1mosaic/cli/ to operate sen2mosaic from the command line.')
    
