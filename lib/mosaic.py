#!/usr/bin/env python

import argparse
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

import utilities

import pdb


global scenes_tile


def _getBands(resolution):
    '''
    Get bands and resolutions for each
    '''
    
    band_list = []
    res_list = []
    
    if resolution == 60 or resolution == 0:
        band_list.extend(['B01','B02','B03','B04','B05','B06','B07','B8A','B09','B11','B12'])
        res_list.extend([60] * 11)
        
    if resolution == 20 or resolution == 0:
        band_list.extend(['B02','B03','B04','B05','B06','B07','B8A','B11','B12'])
        res_list.extend([20] * 9)
        
    if resolution == 10 or resolution == 0:
        band_list.extend(['B02','B03','B04','B08'])
        res_list.extend([10] * 4)
        
    return np.array(res_list), np.array(band_list)


def _getImageOrder(scenes, image_n):
    '''
    Sort tiles, so that most populated is processed first to improve quality of colour balancing
    
    Args:
        scenes: A list of level 2A inputs (of class LoadScene).
        image_n: An array of integers from generateSCLArray(), which describes the scene that each pixel should come from. 0 = No data, 1 = first scene, 2 = second scene etc.
    '''
    
    def _getCentre(scene, ref_scene):
        '''
        Function to get centre point from scene in CRS of ref_scene
        '''
        from osgeo import osr
        
        # Set up function to translate coordinates from source to destination CRS for each scene
        tx = osr.CoordinateTransformation(ref_scene.metadata.proj, scene.metadata.proj)
        
        # And translate the source coordinates
        x_min, y_min, z = tx.TransformPoint(scene.metadata.extent[0], scene.metadata.extent[1])
        x_max, y_max, z = tx.TransformPoint(scene.metadata.extent[2], scene.metadata.extent[3])
        
        return ((y_max - y_min) / 2.) + y_min, ((x_max - x_min) / 2.) + x_min
        
        
    # tile_count = Number of included pixels by tile
    # tile_name  = Granule name in format T##XXX
    # tile_total = Total number of pixels from all images of each tile_count
    # tile_dist  = Euclidean distance to reference tile (in ref tile units)
    
    num, count = np.unique(image_n[image_n!=0], return_counts = True)
    num_sorted = zip(*sorted(zip(count,num), reverse = True))[1]
    tile_count = np.zeros(len(scenes), dtype=np.int)
    tile_count[num-1] = count
    
    tile_name = np.array([scene.tile for scene in scenes])
    
    tile_total = np.zeros(len(scenes),dtype=np.int)
    for tile in np.unique(tile_name):
        tile_total[tile_name == tile] = np.sum(tile_count[tile_name == tile])
    
    # Get reference scene
    ref_scene = np.array(scenes)[tile_total==np.max(tile_total)][0]
    
    tile_dist = []
    
    ref_y, ref_x = _getCentre(ref_scene, ref_scene)
    for n, scene in enumerate(scenes):
        y, x = _getCentre(scene, ref_scene)
        tile_dist.append((((y - ref_y)**2) + ((x - ref_x)**2)) ** 0.5)
    tile_dist = np.array(tile_dist)
    
    # Sort first by distance to referenece tile, then tile name (some tiles are equidistant), then then by contribution of pixels from each overpass. This improves the quality of colour balancing.
    tile_number = np.lexsort((tile_count, tile_name, tile_dist*-1))[::-1] + 1
    
    # Exclude tiles where no data are used
    tile_number = tile_number[tile_count[tile_number-1] > 0]
    
    return tile_number


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


def _makeBlocks(band, scene, step = 2000, percentile = 25., cloud_buffer = 0, masked_vals = [], temp_dir = '/tmp'):
    '''
    Function to build a series of blocks of size (step, step) to enable multiprocessing and prevent overloading memory
    
    Args:
        band: Image band (e.g. 'B02')
        scene: An example Sentinel-2 LoadScene() object
        step: Step size, an integer determining block size
        percentile: Percentile of reflectance to take from valid pixels
        cloud_buffer: Buffer to place around cloud pixels in meters
    
    Returns:
        A list of blocks to be processed by _doComposite()
    '''
    
    blocks = []
    
    for col in range(0, scene.metadata.ncols, step):
        col_step = step if col + step <= scene.metadata.ncols else scene.metadata.ncols - col
        for row in range(0, scene.metadata.nrows, step):
            row_step = step if row + step <= scene.metadata.nrows else scene.metadata.nrows - row
            if row_step ==0 or col_step ==0: pdb.set_trace()
            blocks.append([band, col, col_step, row, row_step, percentile, cloud_buffer, masked_vals, temp_dir])
     
    return blocks

def _doComposite(input_list):
    '''
    Function to build a cloud-free composite image for a block based on a percentile of surface reflectance.
    Internal function for buildMosaic.
    
    Args:
        input_list: A list containing band, col, col_step, row, row_step from _makeBlocks(), percentile, cloud_buffer, and masked_vals 
    Returns:
        A composite image for input block
    '''
    
    band, col, col_step, row, row_step, percentile, cloud_buffer, masked_vals, temp_dir = input_list

    # Mask stack
    m = np.zeros((len(scenes_tile), col_step, row_step), dtype = np.uint8)
    b = np.zeros((len(scenes_tile), col_step, row_step), dtype = np.float32)
    
    for n, scene in enumerate(scenes_tile):
        
        m[n,:,:] = scene.getMask(correct = True, chunk = [row,col,row_step,col_step], cloud_buffer = cloud_buffer, temp_dir = temp_dir)
        
        if m[n,:,:] .sum() == 0: continue
        
        b[n,:,:] = scene.getBand(band, chunk = [row,col,row_step,col_step])
    
    # If nodata in the entire chunk, skip processing
    if m.sum() == 0: return np.zeros_like(b).astype(np.uint16), np.zeros_like(b).astype(np.uint8)
    
    bm = np.ma.array(b, mask = np.ones_like(m,dtype=np.bool))
    
    # Add pixels in order of desirability
    nodata = np.ones_like(b[0,:,:], dtype = np.bool)
    slc = np.zeros_like(b[0,:,:], dtype = np.uint8)
    slc_count = np.zeros_like(b[0,:,:], dtype = np.uint8)
    slc_assigned = np.zeros_like(b[0,:,:], dtype = np.bool)

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
       

def buildMosaic(scenes, band, md_dest, output_dir = os.getcwd(), output_name = 'mosaic', step = 2000, cloud_buffer = 0, processes = 1, percentile = 25., colour_balance = False, masked_vals = [0,1,2,3,7,8,9,10,11], output_mask = True, temp_dir = '/tmp', verbose = False, resampling = 0):
    """
    """
        
    global scenes_tile
    
    for m in masked_vals:
        assert type(m) == int, "Masked values must all be integers."
        
    # Sort scenes for tidiness
    scenes_sorted = utilities.sortScenes(scenes)
        
    composite_out = md_dest.createBlankArray(dtype = np.uint16)
    slc_out = md_dest.createBlankArray(dtype = np.uint8)
    
    # Process one Sentinel-2 tile at a time
    for tile in np.unique([s.tile for s in scenes_sorted]):
        
        scenes_tile = np.array(scenes_sorted)[np.array([s.tile for s in scenes_sorted]) == tile]
        
        scene = scenes_tile[0]
                
        composite = np.zeros((scene.metadata.ncols, scene.metadata.nrows), dtype = np.uint16)
        slc = np.zeros((scene.metadata.ncols, scene.metadata.nrows), dtype = np.uint8)
        
        blocks = _makeBlocks(band, scene, step = step, percentile = percentile, cloud_buffer = cloud_buffer, masked_vals = masked_vals, temp_dir = temp_dir)        
        
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
        composite_rep = utilities.reprojectBand(scene, composite, md_dest, dtype = 3, resampling = resampling)
        slc_rep = utilities.reprojectBand(scene, slc, md_dest, dtype = 1, resampling = 0)
        
        # Do optional colour balancing
        if colour_balance:
            composite_rep = utilities.colourBalance(np.ma.array(composite_rep, mask = composite_rep == 0), np.ma.array(composite_out, mask = composite_out == 0), verbose = verbose)
            composite_rep = composite_rep.filled(0)
        
        # Add pixels to the output mosaic
        sel = composite_rep!=0
        composite_out[sel] = composite_rep[sel]
        slc_out[sel] = slc_rep[sel]
    
    # Output composite image
    utilities.createGdalDataset(md_dest, data_out = composite_out, filename = '%s/%s_R%sm_%s.tif'%(output_dir, output_name, str(scene.resolution), band), driver='GTiff', nodata = 0, options = ['COMPRESS=LZW'])        
    
    # Output mask
    if output_mask:
        utilities.createGdalDataset(md_dest, data_out = slc_out, filename = '%s/%s_R%sm_SLC.tif'%(output_dir, output_name, str(scene.resolution)), driver='GTiff', options = ['COMPRESS=LZW'])        
    
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


def main(source_files, extent_dest, EPSG_dest, start = '20150101', end = datetime.datetime.today().strftime('%Y%m%d'), resolution = 0, correct_mask = True, cloud_buffer = 0., colour_balance = False, processes = 1, output_dir = os.getcwd(), output_name = 'mosaic', masked_vals = 'auto', temp_dir = '/tmp', verbose = False):
    """main(source_files, extent_dest, EPSG_dest, start = '20150101', end = datetime.datetime.today().strftime('%Y%m%d'), resolution = 0, correct_mask = True, cloud_buffer = 0., colour_balance = False, processes = 1, output_dir = os.getcwd(), output_name = 'mosaic', verbose = False):
    
    Function to generate seamless mosaics from a list of Sentinel-2 level-2A input files.
        
    Args:
        source_files: A list of level 3A input files.
        extent_dest: List desciribing corner coordinate points in destination CRS [xmin, ymin, xmax, ymax].
        EPSG_dest: EPSG code of destination coordinate reference system. Must be a UTM projection. See: https://www.epsg-registry.org/ for codes.
        start: Start date to process, in format 'YYYYMMDD' Defaults to start of Sentinel-2 era.
        end: End date to process, in format 'YYYYMMDD' Defaults to today's date.
        resolution: Process 10, 20, or 60 m bands. Defaults to processing all three.
        correct_mask: Set True to apply improvements to mask from sen2cor.
        processes: Number of processes to run similtaneously. Defaults to 1.
        output_dir: Optionally specify an output directory.
        output_name: Optionally specify a string to precede output file names. Defaults to 'mosaic'.
        masked_vals: List of SLC mask values to not include in the final mosaic. Defaults to 'auto', which masks everything except [4,5,6]
        temp_dir: Directory to temporarily write L1C mask files. Defaults to /tmp
        verbose: Make script verbose (set True).
    """
    
    assert len(extent_dest) == 4, "Output extent must be specified in the format [xmin, ymin, xmax, ymax]"
    assert extent_dest[0] < extent_dest[2], "Output extent incorrectly specified: xmin must be lower than xmax."
    assert extent_dest[1] < extent_dest[3], "Output extent incorrectly specified: ymin must be lower than ymax."
    assert len(source_files) >= 1, "No source files in specified location."
    assert resolution in [0, 10, 20, 60], "Resolution must be 10, 20, or 60 m, or 0 to process all three."
    assert type(correct_mask) == bool, "correct_mask can only be set to True or False."
    
    # Test that output directory is writeable
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    assert os.path.exists(output_dir), "Output directory (%s) does not exist."%output_dir
    assert os.access(output_dir, os.W_OK), "Output directory (%s) does not have write permission. Try setting a different output directory, or changing permissions with chmod."%output_dir
        
    if masked_vals == 'auto': masked_vals = [0,9]#[0,1,2,3,7,8,9,10,11]
    if masked_vals == 'none': masked_vals = []
    assert type(masked_vals) == list, "Masked values must be a list of integers, or set to 'auto' or 'none'."
    
    res_list, band_list = _getBands(resolution)
    
    # For each of the input resolutions
    for res in np.unique(res_list)[::-1]:
        
        # Load metadata for all Sentinel-2 datasets
        scenes = []
        for source_file in source_files:
            try:
                scenes.append(utilities.LoadScene(source_file, resolution = res))
            except Exception as e:
                print(e)
                print('WARNING: Error in loading scene %s. Continuing.'%source_file)
        
        assert len(scenes) > 0, "Failed to load any scenes for resolution %sm. Check input scenes."%str(res)
        
        # Build metadata of output object
        md_dest = utilities.Metadata(extent_dest, res, EPSG_dest)
        
        # Reduce the pool of scenes to only those that overlap with output tile
        scenes_reduced = utilities.getSourceFilesInTile(scenes, md_dest, start = start, end = end, verbose = verbose)       
        
        # It's only worth processing a tile if at least one input image is inside tile
        if len(scenes_reduced) == 0:
            print("    No data inside specified output area for resolution %s. make sure you specified your bouding box in the correct order (i.e. xmin ymin xmax ymax) and EPSG code correctly. Continuing."%str(res))
            continue
        
        
        output_mask = True
        for band in band_list[res_list==res]:
            
            if verbose: print('Building band %s at %s m resolution'%(band, str(res)))
            
            band_out, QA_out = buildMosaic(scenes_reduced, band, md_dest, output_dir = output_dir, output_name = output_name, colour_balance = colour_balance, cloud_buffer = cloud_buffer, percentile = 25., processes = processes, step = 2000, masked_vals = masked_vals, output_mask = output_mask, temp_dir = temp_dir, verbose = verbose)            
            
            # Only output mask on first iteration
            output_mask = False
            
        # Build VRT output files for straightforward visualisation
        if verbose: print('Building .VRT images for visualisation')
        
        # Natural colour image (10 m)
        buildVRT('%s/%s_R%sm_B04.tif'%(output_dir, output_name, str(res)), '%s/%s_R%sm_B03.tif'%(output_dir, output_name, str(res)), '%s/%s_R%sm_B02.tif'%(output_dir, output_name, str(res)), '%s/%s_R%sm_RGB.vrt'%(output_dir, output_name, str(res)))

        # Near infrared image. Band at (10 m) has a different format to bands at 20 and 60 m.
        if res == 10:
            buildVRT('%s/%s_R%sm_B08.tif'%(output_dir, output_name, str(res)), '%s/%s_R%sm_B04.tif'%(output_dir, output_name, str(res)), '%s/%s_R%sm_B03.tif'%(output_dir, output_name, str(res)), '%s/%s_R%sm_NIR.vrt'%(output_dir, output_name, str(res)))    
        else:
            buildVRT('%s/%s_R%sm_B8A.tif'%(output_dir, output_name, str(res)), '%s/%s_R%sm_B04.tif'%(output_dir, output_name, str(res)), '%s/%s_R%sm_B03.tif'%(output_dir, output_name, str(res)), '%s/%s_R%sm_NIR.vrt'%(output_dir, output_name, str(res)))
        
    print('Processing complete!')


if __name__ == "__main__":
    
    # Set up command line parser    

    parser = argparse.ArgumentParser(description = "Process Sentinel-2 data to a composite mosaic product to a customisable grid square, based on specified UTM coordinate bounds. Data are output as GeoTiff files for each spectral band, with .vrt files for ease of visualisation.")

    parser._action_groups.pop()
    positional = parser.add_argument_group('positional arguments')
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    
    # Positional arguments
    positional.add_argument('infiles', metavar = 'PATH', type = str, default = [os.getcwd()], nargs = '*', help = 'Sentinel 2 input files (level 1C/2A) in .SAFE format. Specify one or more valid Sentinel-2 .SAFE, a directory containing .SAFE files, a Sentinel-2 tile or multiple granules through wildcards (e.g. *.SAFE/GRANULE/*), or a file containing a list of input files. Leave blank to process files in current working directoy. All granules that match input conditions will be included.')
    
    # Required arguments
    required.add_argument('-te', '--target_extent', nargs = 4, metavar = ('XMIN', 'YMIN', 'XMAX', 'YMAX'), type = float, required = True, help = "Extent of output image tile, in format <xmin, ymin, xmax, ymax>.")
    required.add_argument('-e', '--epsg', metavar = 'EPSG', type=int, required = True, help="EPSG code for output image tile CRS. This must be UTM. Find the EPSG code of your output CRS as https://www.epsg-registry.org/.")
    
    # Optional arguments
    optional.add_argument('-l', '--level', type=str, metavar='1C/2A', default = '2A', help = "Input image processing level, '1C' or '2A'. Defaults to '2A'.")
    optional.add_argument('-st', '--start', type = str, default = '20150101', help = "Start date for tiles to include in format YYYYMMDD. Defaults to processing all dates.")
    optional.add_argument('-en', '--end', type = str, default = datetime.datetime.today().strftime('%Y%m%d'), help = "End date for tiles to include in format YYYYMMDD. Defaults to processing all dates.")
    optional.add_argument('-res', '--resolution', metavar = '10/20/60', type=int, default = 0, help="Specify a resolution to process (10, 20, 60, or 0 for all).")
    optional.add_argument('-m', '--masked_vals', metavar = 'N', type=str, nargs='*', default = ['auto'], help="Specify SLC values to not include in the mosaic (e.g. -m 7 8 9). See http://step.esa.int/main/third-party-plugins-2/sen2cor/ for description of sen2cor mask values. Defaults to 'auto', which masks values 0 and 9. Also accepts 'none', to include all values.")
    optional.add_argument('-b', '--colour_balance', action='store_true', default = False, help = "Perform colour balancing between tiles. Defaults to False. Not generally recommended, particularly where working over large areas.")
    optional.add_argument('-c', '--cloud_buffer', type=int, metavar = 'M', default = 0, help = "Apply improvements to sen2cor cloud mask by applying a buffer around cloudy pixels (in meters). Not generally recommended, except where a very conservative mask is desired. Defaults to no buffer.")
    optional.add_argument('-t', '--temp_dir', type=str, metavar = 'DIR', default = '/tmp', help="Directory to write temporary files, only required for L1C data. Defaults to '/tmp'.")
    optional.add_argument('-o', '--output_dir', type=str, metavar = 'DIR', default = os.getcwd(), help="Specify an output directory. Defaults to the present working directory.")
    optional.add_argument('-n', '--output_name', type=str, metavar = 'NAME', default = 'mosaic', help="Specify a string to precede output filename. Defaults to 'mosaic'.")
    optional.add_argument('-p', '--n_processes', type = int, metavar = 'N', default = 1, help = "Specify a maximum number of tiles to process in paralell. Bear in mind that more processes will require more memory. Defaults to 1.")
    optional.add_argument('-v', '--verbose', action='store_true', default = False, help = "Make script verbose.")

    # Get arguments
    args = parser.parse_args()
        
    assert args.level in ['1C', '2A'], "Input level much be '1C' or '2A'."
    
    # Convert masked_vals to integers, where specified
    if args.masked_vals != ['auto'] and args.masked_vals != ['none']:
        masked_vals = [int(m) for m in args.masked_vals]
    else:
        masked_vals = args.masked_vals[0]
    
    # Get absolute path of input .safe files.
    infiles = sorted([os.path.abspath(i) for i in args.infiles])

    infiles = glob.glob(f"{infiles[0]}\\*")

    # Find all matching granule files
    infiles = utilities.prepInfiles(infiles, args.level)
    
    main(infiles, args.target_extent, args.epsg, resolution = args.resolution, start = args.start, end = args.end, cloud_buffer = args.cloud_buffer, colour_balance = args.colour_balance, processes = args.n_processes, output_dir = args.output_dir, output_name = args.output_name, masked_vals = masked_vals, temp_dir = args.temp_dir, verbose = args.verbose)
