#!/usr/bin/env python

import datetime
import glob
import numpy as np
import os
import pandas
import re
import time
import sentinelsat
import zipfile

import pdb


#################################################
### Functions for downloading Sentinel-2 data ###
#################################################

def _removeZip(zip_file):
    """
    Deletes Level 1C .zip file from disk.
    
    Args:
        zip_file: A Sentinel-2 level 1C .zip file from Copernicus Open Access Data Hub.
    """
    
    assert '_MSIL1C_' in zip_file, "removeZip function should only be used to delete Sentinel-2 level 1C compressed .SAFE files"
    assert zip_file.split('/')[-1][-4:] == '.zip', "removeL1C function should only be used to delete Sentinel-2 level 1C compressed .SAFE files"
    
    os.remove(zip_file)
    

def connectToAPI(username, password):
    '''
    Connect to the SciHub API with sentinelsat.
    
    Args:
        username: Scihub username. Sign up at https://scihub.copernicus.eu/.
        password: Scihub password.        
    '''
        
    # Let API be accessed by other functions
    global scihub_api
    
    # Disconnect from any previous session
    scihub_api = None
    
    # Connect to Sentinel API
    scihub_api = sentinelsat.SentinelAPI(username, password, 'https://scihub.copernicus.eu/dhus')
    

def _get_filesize(products_df):
    """
    Extracts file size in MB from a Sentinel products pandas dataframe.
    
    Args:
        products_df: A pandas dataframe from search().
    Returns:
        A numpy array with file sizes in MB.
    """
    
    size = [int(float(str(i).split(' ')[0])) for  i in products_df['size'].values]
    suffix = [str(i).split(' ')[1].lower() for  i in products_df['size'].values]
    
    size_mb = []
    
    for this_size, this_suffix in zip(size, suffix):
        if this_suffix == 'kb' or this_suffix == 'kib':
            size_mb.append(this_size * 0.001)
        elif this_suffix == 'mb' or this_suffix == 'mib':
            size_mb.append(this_size * 1.)
        elif this_suffix == 'gb' or this_suffix == 'gib':
            size_mb.append(this_size * 1000.)
        else:
            size_mb.append(this_size * 0.000001)

    return np.array(size_mb)


def search(tile, level = '1C', start = '20150523', end = datetime.datetime.today().strftime('%Y%m%d'),  maxcloud = 100, minsize = 25.):
    """search(tile, start = '20161206', end = datetime.datetime.today().strftime('%Y%m%d'),  maxcloud = 100, minsize_mb = 25.)
    
    Searches for images from a single Sentinel-2 Granule that meet conditions of date range and cloud cover.
    
    Args:
        tile: A string containing the name of the tile to to download.
        level: Download level '1C' (default) or '2A' data.
        start: Start date for search in format YYYYMMDD. Defaults to 20150523.
        end: End date for search in format YYYYMMDD. Defaults to today's date.
        maxcloud: An integer of maximum percentage of cloud cover to download. Defaults to 100 %% (download all images, regardless of cloud cover).
        minsize: A float with the minimum filesize to download in MB. Defaults to 25 MB.  Be aware, file sizes smaller than this can result sen2three crashing.
    
    Returns:
        A pandas dataframe with details of scenes matching conditions.
    """
    
    # Test that we're connected to the 
    assert 'scihub_api' in globals(), "The global variable scihub_api doesn't exist. You should run connectToAPI(username, password) before searching the data archive."

    # Validate tile input format for search
    assert bool(re.match("[0-9]{2}[A-Z]{3}$",tile)), "The tile name input (%s) does not match the format ##XXX (e.g. 36KWA)."%tile
    
    assert level in ['1C', '2A'], "Level must be '1C' or '2A'."
    
    # Set up start and end dates
    startdate = sentinelsat.format_query_date(start)
    enddate = sentinelsat.format_query_date(end)
    
    # Search data, filtering by options.
    products = scihub_api.query(beginposition = (startdate,enddate),
                         platformname = 'Sentinel-2',
                         producttype = 'S2MSI%s'%level,
                         cloudcoverpercentage = (0,maxcloud),
                         filename = '*T%s*'%tile)
    
    # convert to Pandas DataFrame, which can be searched modified before commiting to download()
    products_df = scihub_api.to_dataframe(products)
    
    # Where no results for tile    
    if len(products_df) == 0: return products_df
    
    products_df['filesize_mb'] = _get_filesize(products_df)
    
    products_df = products_df[products_df['filesize_mb'] >= float(minsize)]
    # print('Found %s matching images'%str(len(products_df)))
     
    return products_df


def download(products_df, output_dir = os.getcwd()):
    ''' download(products_df, output_dir = os.getcwd())
    
    Downloads all images from a dataframe produced by sentinelsat.
    
    Args:
        products_df: Pandas dataframe from search() function.
        output_dir: Optionally specify an output directory. Defaults to the present working directory.
    '''
    
    assert os.path.isdir(output_dir), "Output directory doesn't exist."
    
    if products_df.empty == True:
        print('WARNING: No products found to download. Check your search terms.')
        raise
        
    else:
        
        downloaded_files = []
        
        for uuid, filename in zip(products_df['uuid'], products_df['filename']):
            
            if os.path.exists('%s/%s'%(output_dir, filename)):
                print('Skipping file %s, as it has already been downloaded in the directory %s. If you want to re-download it, delete it and run again.'%(filename, output_dir))
            
            elif os.path.exists('%s/%s'%(output_dir, filename[:-5] + '.zip')):
                
                print('Skipping file %s, as it has already been downloaded and extracted in the directory %s. If you want to re-download it, delete it and run again.'%(filename, output_dir))
            
            elif scihub_api.get_product_odata(uuid)['Online'] == False:
                
                print('Skipping file %s, as it is part of the long term archive. Consider ordering directly from the Copernicus Open Access Hub.'%filename)
                
            else:
                
                # Download selected product
                print('Downloading %s...'%filename)
                scihub_api.download(uuid, output_dir)
                
                downloaded_files.append(('%s/%s'%(output_dir.rstrip('/'), filename)).replace('.SAFE','.zip'))
    
    return downloaded_files


def decompress(zip_files, output_dir = os.getcwd(), remove = False):
    '''decompress(zip_files, output_dir = os.getcwd(), remove = False
    
    Decompresses .zip files downloaded from SciHub, and optionally removes original .zip file.
    
    Args:
        zip_files: A list of .zip files to decompress.
        output_dir: Optionally specify an output directory. Defaults to the present working directory.
        remove: Boolean value, which when set to True deletes level 1C .zip files after decompression is complete. Defaults to False.
    '''

    if type(zip_files) == str: zip_files = [zip_files]
    
    for zip_file in zip_files:
        assert zip_file[-4:] == '.zip', "Files to decompress must be .zip format."
    
    # Decompress each zip file
    for zip_file in zip_files:
        
        # Skip those files that have already been extracted
        if os.path.exists('%s/%s'%(output_dir, zip_file.split('/')[-1].replace('.zip', '.SAFE'))):
            print('Skipping extraction of %s, as it has already been extracted in directory %s. If you want to re-extract it, delete the .SAFE file.'%(zip_file, output_dir))
        
        else:     
            print('Extracting %s'%zip_file)
            with zipfile.ZipFile(zip_file) as obj:
                obj.extractall(output_dir)
            
            # Delete zip file
            if remove: _removeZip(zip_file)

if __name__ == '__main__':
    '''
    '''
        
    print('The sen2mosaic command line interface has been moved! Please use scripts in .../sen2mosaic/cli/ to operate sen2mosaic from the command line.')
    
