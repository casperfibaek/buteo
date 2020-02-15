#!/usr/bin/env python

import glob
import numpy as np
import os
import psutil
import re
from scipy import ndimage
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET

import sen2mosaic.multiprocess

import pdb

#################################################################
### Functions for preprocessing of Sentinel-2 L1C data to L2A ###
#################################################################

def _which(program):
    '''
    Tests whether command line script exists. Mimics which in command line.
    From: https://stackoverflow.com/questions/377017/test-if-executable-exists-in-python
    
    Args:
        program: String to be executed
    Returns:
        Path to script, or None where program does not exist.
    '''
    
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


### Primary functions

def _setGipp(gipp, median_filter = 0, v255 = False):
    """
    Function that tweaks options in sen2cor's L2A_GIPP.xml file to specify an output directory.
    
    Args:
        gipp: The path to a copy of the L2A_GIPP.xml file.
        median_filter: Set 0-3 to perform smoothing operation on classified scene. Not currently used.
    Returns:
        The directory location of a temporary .gipp file, for input to L2A_Process
    """
    
    # Test that GIPP and output directory exist
    assert gipp != None, "GIPP file must be specified if you're changing sen2cor options."
    assert os.path.isfile(gipp), "GIPP XML options file doesn't exist at the location %s."%gipp  
    assert median_filter in [0, 1, 2, 3], "median_filter can only be 0-3."
       
    # Read GIPP file
    tree = ET.ElementTree(file = gipp)
    root = tree.getroot()
    
    # Change output directory (if old version)
    if v255: root.find('Common_Section/Target_Directory').text = output_dir
        
    root.find('Scene_Classification/Filters/Median_Filter').text = str(median_filter)
    
    # Generate a temporary output file
    temp_gipp = tempfile.mktemp(suffix='.xml')
        
    # Ovewrite old GIPP file with new options
    tree.write(temp_gipp)
    
    return temp_gipp


def getL2AFilename(L1C_file, output_dir = os.getcwd(), SAFE = False):
    """
    Determine the level 2A tile path name from an input file (level 1C) tile.
    
    Args:
        L1C_file: Input level 1C .SAFE file tile (e.g. '/PATH/TO/*.SAFE/GRANULE/*').
        output_dir: Directory of processed file.
        SAFE: Return path of base .SAFE file
    Returns:
        The name and directory of the output file
    """
    
    # Determine output file name, replacing two instances only of substring L1C_ with L2A_    
    outfile = re.sub(r"L1C_","L2A_", L1C_file)
    
    # Allow for changes in file format
    outfile = re.sub(r"_N[0-9]{4}","_N????", outfile)
    
    # Replace _OPER_ with _USER_ for case of old file format (in final 2 cases)
    outfile = outfile[::-1].replace('_OPER_'[::-1],'_USER_'[::-1],2)[::-1]
    
    # Replace processing date
    outfile = re.sub(r"_[0-9]{8}T[0-9]{6}.SAFE","_????????T??????.SAFE", outfile)
    
    outpath = os.path.join(output_dir, outfile)
    
    # Get outpath of base .SAFE file
    if SAFE: outpath = '/'.join(outpath.split('.SAFE')[:-1]) + '.SAFE'# '/'.join(outpath.split('/')[:-2])
    
    return outpath.rstrip('/')


def processToL2A(granule, gipp = None, output_dir = os.getcwd(), resolution = 0, sen2cor = 'L2A_Process', sen2cor_255 = None, product_format = 'SAFE_COMPACT', verbose = False):
    """
    Processes Sentinel-2 level 1C files to level L2A with sen2cor.
    
    Args:
        granule: A level 1C Sentinel-2 granule file.
        gipp: Optionally specify a copy of the L2A_GIPP.xml file in order to tweak options.
        output_dir: Optionally specify an output directory. Defaults to current working directory.
        resolution: Optionally specify a resolution (10, 20 or 60) meters. Defaults to 0, which processes all three
        sen2cor: Location of sen2cor (v2.8). This version of sen2cor only supports the newer 'SAFE_COMPACT' file format. Defaults to 'L2A_Process'
        sen2cor_255: Location of sen2cor (v2.5.5). Only required if processing old 'SAFE' format files.
        verbose: Set True to print progress
        product_format: Either the new ('SAFE_COMPACT') or old ('SAFE') Sentinel-2 format.
    Returns:
        Absolute file path to the output file.
    """
    
    # At present, L2A_Process cannot just process at 10 m resolution
    if resolution == 10: resolution = 0
 
    # Test that input file is in .SAFE format
    assert '_MSIL1C_' in granule.split('/')[-3], "Input files must be in level 1C format, and provided as a path to a granule file."
    
    # Test that product_format exists
    assert product_format in ['SAFE', 'SAFE_COMPACT'], "product_format can only be 'SAFE' or 'SAFE_COMPACT'. %s was specified."%product_format
    
    if product_format == 'SAFE': assert sen2cor_255 != None, "The old format of Sentinel-2 data ('SAFE') can only be preprocessed by the old version of sen2cor (v2.5.5). Input granule (%s) has that old format, so you must also input the path to an old version of sen2cor."%granule
    
    # Get .SAFE filename
    filename = '/'.join(granule.split('/')[:-2])
    
    # Test that resolution is reasonable
    assert resolution in [0, 10, 20, 60], "Input resolution must be 10, 20, 60, or 0 (for all resolutions). The input resolution was %s"%str(resolution)
    
    # Test that sen2cor is installed
    assert _which(sen2cor) is not None, "Can't find program sen2cor given command (%s). Ensure that sen2cor has been correctly installed and that it's path has been specified correctly."%str(sen2cor)
    
    if sen2cor_255 is not None: assert _which(sen2cor_255) is not None, "Can't find program sen2cor (v2.5.5) given command (%s). Ensure that sen2cor (v2.5.5) has been correctly installed and that it's path has been specified correctly."%str(sen2cor_255)
    
    # Test that output directory is writeable
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    assert os.path.exists(output_dir), "Output directory (%s) does not exist."%output_dir
    assert os.access(output_dir, os.W_OK), "Output directory (%s) does not have write permission. Try setting a different output directory"%output_dir
    
    # Determine output filename
    outpath = getL2AFilename(granule, output_dir = output_dir)
    
    # Check if output file already exists
    if len(glob.glob(outpath)) > 0:
      print('The output file %s already exists! Delete it to run sen2cor.'%outpath)
      return outpath
    
    # Get location of exemplar gipp file for modification
    if gipp == None:
        if product_format == 'SAFE_COMPACT':
            gipp = '/'.join(os.path.abspath(__file__).split('/')[:-2] + ['cfg','L2A_GIPP.xml'])
        else:
            gipp = '/'.join(os.path.abspath(__file__).split('/')[:-2] + ['cfg','L2A_GIPP_v255.xml'])
            
    # Base command, including GIPP file appropriately set up
    if product_format == 'SAFE_COMPACT':
        temp_gipp = _setGipp(gipp, median_filter = 0, v255 = False)
        command = [sen2cor, '--GIP_L2A', temp_gipp]
    else:
        temp_gipp = _setGipp(gipp, median_filter = 0, v255 = True)
        command = [sen2cor_255, '--GIP_L2A', temp_gipp]
    
    # Specify resolution
    if resolution != 0: command.extend(['--resolution', str(resolution)])
    
    # Specify output directory
    if product_format == 'SAFE_COMPACT': command.extend(['--output_dir', output_dir])
    
    # Add input filename
    if product_format == 'SAFE_COMPACT':
        command.extend([filename])
    else:
        command.extend([granule])
    
    # print(command for user info
    if verbose: print(' '.join(command))
    
    # Do the processing, and capture exceptions
    try:
        output_text = sen2mosaic.multiprocess.runCommand(command, verbose = verbose)
        t=1
    except Exception as e:
        # Tidy up temporary options file
        os.remove(temp_gipp)
        raise
    
    # Tidy up temporary options file
    os.remove(temp_gipp)
    
    # Get path of .SAFE file.
    outpath_SAFE = getL2AFilename(granule, output_dir = output_dir, SAFE = True)
        
    # Occasionally sen2cor outputs a _null directory. This can cause problems, so should be removed.
    bad_directories = glob.glob('%s/GRANULE/*_null/'%outpath_SAFE)
    
    if bad_directories:
        [shutil.rmtree(bd) for bd in bad_directories]
     
    return outpath


def testCompletion(L1C_file, output_dir = os.getcwd(), resolution = 0):
    """
    Test for successful completion of sen2cor processing. 
    
    Args:
        L1C_file: Path to level 1C granule file (e.g. /PATH/TO/*_L1C_*.SAFE/GRANULE/*)
    Returns:
        A boolean describing whether processing completed sucessfully.
    """
      
    L2A_file = getL2AFilename(L1C_file, output_dir = output_dir, SAFE = False)
    
    failure = False
    
    # Test all expected 10 m files are present
    if resolution == 0 or resolution == 10:
        
        for band in ['B02', 'B03', 'B04', 'B08', 'AOT', 'TCI', 'WVP']:
            
            if not len(glob.glob('%s/IMG_DATA/R10m/*_%s_10m.jp2'%(L2A_file,band))) == 1:
                failure = True
    
    # Test all expected 20 m files are present
    if resolution == 0 or resolution == 20:
        
        for band in ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'AOT', 'TCI', 'WVP', 'SCL']:
            
            if not len(glob.glob('%s/IMG_DATA/R20m/*_%s_20m.jp2'%(L2A_file,band))) == 1:
                
                failure = True

    # Test all expected 60 m files are present
    if resolution == 0 or resolution == 60:
        
        for band in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'AOT', 'TCI', 'WVP', 'SCL']:
            
            if not len(glob.glob('%s/IMG_DATA/R60m/*_%s_60m.jp2'%(L2A_file,band))) == 1:
                
                failure = True
    
    # At present we only report failure/success, can be extended to type of failure 
    return failure == False
    
    
if __name__ == '__main__':
    '''
    '''
        
    print('The sen2mosaic command line interface has been moved! Please use scripts in .../sen2mosaic/cli/ to operate sen2mosaic from the command line.')
    
