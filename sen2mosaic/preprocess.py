#!/usr/bin/env python

import argparse
import functools
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

import multiprocess
import utilities

import pdb



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


def _setGipp(gipp, output_dir = os.getcwd(), n_processes = 1):
    """
    Function that tweaks options in sen2cor's L2A_GIPP.xml file to specify an output directory.
    
    Args:
        gipp: The path to a copy of the L2A_GIPP.xml file.
        output_dir: The desired output directory. Defaults to the same directory as input files.
        n_processes: The number of processes to use for sen2cor. Defaults to 1.
    
    Returns:
        The directory location of a temporary .gipp file, for input to L2A_Process
    """
    
    # Test that GIPP and output directory exist
    assert gipp != None, "GIPP file must be specified if you're changing sen2cor options."
    assert os.path.isfile(gipp), "GIPP XML options file doesn't exist at the location %s."%gipp  
    assert os.path.isdir(output_dir), "Output directory %s doesn't exist."%output_dir
    
    # Adds a trailing / to output_dir if not already specified
    output_dir = os.path.join(output_dir, '')
   
    # Read GIPP file
    tree = ET.ElementTree(file = gipp)
    root = tree.getroot()
    
    # Change output directory    
    root.find('Common_Section/Target_Directory').text = output_dir
    
    # Change number of processes
    root.find('Common_Section/Nr_Processes').text = str(n_processes)
    
    # Generate a temporary output file
    temp_gipp = tempfile.mktemp(suffix='.xml')
        
    # Ovewrite old GIPP file with new options
    tree.write(temp_gipp)
    
    return temp_gipp



def getL2AFile(L1C_file, output_dir = os.getcwd(), SAFE = False):
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
    outfile = '/'.join(L1C_file.split('/')[-3:])[::-1].replace('L1C_'[::-1],'L2A_'[::-1],2)[::-1]
    
    # Replace _OPER_ with _USER_ for case of old file format (in final 2 cases)
    outfile = outfile[::-1].replace('_OPER_'[::-1],'_USER_'[::-1],2)[::-1]
    
    outpath = os.path.join(output_dir, outfile)
    
    # Get outpath of base .SAFE file
    if SAFE: outpath = '/'.join(outpath.split('.SAFE')[:-1]) + '.SAFE'# '/'.join(outpath.split('/')[:-2])
    
    return outpath.rstrip('/')


def processToL2A(infile, gipp = None, output_dir = os.getcwd(), n_processes = 1, resolution = 0, verbose = False):
    """
    Processes Sentinel-2 level 1C files to level L2A with sen2cor.
    
    Args:
        infile: A level 1C Sentinel-2 .SAFE file.
        gipp: Optionally specify a copy of the L2A_GIPP.xml file in order to tweak options.
        output_dir: Optionally specify an output directory. Defaults to current working directory.
        n_processes: Number of processes to allocate to sen2cor. We don't use this, as we implement our own paralellisation via multiprocessing.
        resolution: Optionally specify a resolution (10, 20 or 60) meters. Defaults to 0, which processes all three
    Returns:
        Absolute file path to the output file.
    """
    
    # At present, L2A_Process cannot just process at 10 m resolution
    if resolution == 10: resolution = 0
 
    # Test that input file is in .SAFE format
    assert '_MSIL1C_' in infile.split('/')[-3], "Input files must be in level 1C format."
    
    # Test that resolution is reasonable
    assert resolution in [0, 10, 20, 60], "Input resolution must be 10, 20, 60, or 0 (for all resolutions). The input resolution was %s"%str(resolution)
    
    # Test that sen2cor is installed
    assert _which('L2A_Process') is not None, "Can't find program L2A_Process. Ensure that sen2cor has been correctly installed."
    
    # Test that output directory is writeable
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    assert os.path.exists(output_dir), "Output directory (%s) does not exist."%output_dir
    assert os.access(output_dir, os.W_OK), "Output directory (%s) does not have write permission. Try setting a different output directory"%output_dir
    
    # Determine output filename
    outpath = getL2AFile(infile, output_dir = output_dir)
    
    # Check if output file already exists
    if os.path.exists(outpath):
      print('The output file %s already exists! Delete it to run L2_Process.'%outpath)
      return outpath
    
    # Get location of exemplar gipp file for modification
    if gipp == None:
        gipp = '/'.join(os.path.abspath(__file__).split('/')[:-2] + ['cfg','L2A_GIPP.xml'])
        
    # Set options in L2A GIPP xml. Returns the modified .GIPP file. This prevents concurrency issues in multiprocessing.
    temp_gipp = _setGipp(gipp, output_dir = output_dir, n_processes = n_processes)
             
    # Set up sen2cor command
    if resolution != 0:
        command = ['L2A_Process', '--GIP_L2A', temp_gipp, '--resolution', str(resolution), infile]
    else:
        command = ['L2A_Process', '--GIP_L2A', temp_gipp, infile]
    
    # print(command for user info
    if verbose: print(' '.join(command))
       
    # Do the processing, and capture exceptions
    try:
        output_text = multiprocess.runCommand(command, verbose = verbose)
    except Exception as e:
        # Tidy up temporary options file
        os.remove(temp_gipp)
        raise
    
    # Tidy up temporary options file
    os.remove(temp_gipp)
    
    # Get path of .SAFE file.
    outpath_SAFE = getL2AFile(infile, output_dir = output_dir, SAFE = True)
    
    # Test if AUX_DATA output directory exists. If not, create it, as it's absense crashes sen2three.
    if not os.path.exists('%s/AUX_DATA'%outpath_SAFE):
        os.makedirs('%s/AUX_DATA'%outpath_SAFE)
    
    # Occasionally sen2cor outputs a _null directory. This needs to be removed, or sen2Three will crash.
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
      
    L2A_file = getL2AFile(L1C_file, output_dir = output_dir, SAFE = False)
    
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



def main(infile, gipp = None, output_dir = os.getcwd(), resolution = 0, verbose = False):
    """
    Function to initiate sen2cor on level 1C Sentinel-2 files and perform improvements to cloud masking. This is the function that is initiated from the command line.
    
    Args:
        infile: A Level 1C Sentinel-2 .SAFE file.
        gipp: Optionally specify a copy of the L2A_GIPP.xml file in order to tweak options.
        output_dir: Optionally specify an output directory. The option gipp must also be specified if you use this option.
    """
    
    if verbose: print('Processing %s'%infile.split('/')[-1])
    
    try:
        L2A_file = processToL2A(infile, gipp = gipp, output_dir = output_dir, resolution = resolution, verbose = verbose)
    except Exception as e:
        raise
                
    # Test for completion, and report back
    if testCompletion(infile, output_dir = output_dir, resolution = resolution) == False:    
        print('WARNING: %s did not complete processing.'%infile)
    

if __name__ == '__main__':
    '''
    '''
    
    
    # Set up command line parser
    parser = argparse.ArgumentParser(description = 'Process level 1C Sentinel-2 data from the Copernicus Open Access Hub to level 2A. This script initiates sen2cor, which performs atmospheric correction and generate a cloud mask. This script also performs simple improvements to the cloud mask.')
    
    parser._action_groups.pop()
    required = parser.add_argument_group('Required arguments')
    positional = parser.add_argument_group('Positional arguments')
    optional = parser.add_argument_group('Optional arguments')
    
    # Required arguments
    
    # Optional arguments
    positional.add_argument('infiles', metavar = 'L1C_FILES', type = str, default = [os.getcwd()], nargs = '*', help = 'Sentinel 2 input files (level 1C) in .SAFE format. Specify one or more valid Sentinel-2 .SAFE, a directory containing .SAFE files, a Sentinel-2 tile or multiple granules through wildcards (e.g. *.SAFE/GRANULE/*), or a file containing a list of input files. Leave blank to process files in current working directoy. All granules that match input conditions will be atmospherically corrected.')
    optional.add_argument('-t', '--tile', type = str, default = '', help = 'Specify a specific Sentinel-2 tile to process. If omitted, all tiles in L1C_FILES will be processed.')
    optional.add_argument('-g', '--gipp', type = str, default = None, help = 'Specify a custom L2A_Process settings file (default = sen2cor/cfg/L2A_GIPP.xml).')
    optional.add_argument('-o', '--output_dir', type = str, metavar = 'DIR', default = os.getcwd(), help = "Specify a directory to output level 2A files. If not specified, atmospherically corrected images will be written to the same directory as input files.")
    optional.add_argument('-res', '--resolution', type = int, metavar = '10/20/60', default = 0, help = "Process only one of the Sentinel-2 resolutions, with options of 10, 20, or 60 m. Defaults to processing all three. N.B It is not currently possible to only the 10 m resolution, an input of 10 m will instead process all resolutions.")
    optional.add_argument('-p', '--n_processes', type = int, metavar = 'N', default = 1, help = "Specify a maximum number of tiles to process in paralell. Bear in mind that more processes will require more memory. Defaults to 1.")
    optional.add_argument('-v', '--verbose', action='store_true', default = False, help = "Make script verbose.")
    
    # Get arguments
    args = parser.parse_args()
        
    # Get all infiles that match tile and file pattern
    infiles = utilities.prepInfiles(args.infiles, '1C', tile = args.tile)
     
    # Get absolute path for output directory
    args.output_dir = os.path.abspath(args.output_dir)
    
    # Strip the output files already exist.
    for infile in infiles[:]:
        outpath = getL2AFile(infile, output_dir = args.output_dir)
        
        if os.path.exists(outpath):
            print('WARNING: The output file %s already exists! Skipping file.'%outpath)
    
    if len(infiles) == 0: raise ValueError('No level 1C Sentinel-2 files detected in input directory that match specification.')
        
    if args.n_processes == 1:
        
        # Keep things simple when using one processor
        for infile in infiles:
            
            main(infile, gipp = args.gipp, output_dir = args.output_dir, resolution = args.resolution, verbose = args.verbose) 
    
    else:

        # Set up function with multiple arguments, and run in parallel
        main_partial = functools.partial(main, gipp = args.gipp, output_dir = args.output_dir, resolution = args.resolution, verbose = args.verbose)
    
        multiprocess.runWorkers(main_partial, args.n_processes, infiles)
    
    # Test for completion
    completion = np.array([testCompletion(infile, output_dir = args.output_dir, resolution = args.resolution) for infile in infiles])
    
    # Report back
    if completion.sum() > 0: print('Successfully processed files:')
    for infile in np.array(infiles)[completion == True]:
        print(infile)
    if (completion == False).sum() > 0: print('Files that failed:')
    for infile in np.array(infiles)[completion == False]:
        print(infile)

    
    

