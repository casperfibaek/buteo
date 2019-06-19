#!/usr/bin/env python

import argparse
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


def _removeZip(zip_file):
    """
    Deletes Level 1C .zip file from disk.
    
    Args:
        #A Sentinel-2 level 1C .zip file from Copernicus Open Access Data Hub.
    """
    
    assert '_MSIL1C_' in zip_file, "removeZip function should only be used to delete Sentinel-2 level 1C compressed .SAFE files"
    assert zip_file.split('/')[-1][-4:] == '.zip', "removeL1C function should only be used to delete Sentinel-2 level 1C compressed .SAFE files"
    
    os.remove(zip_file)
    

def _validateTile(tile):
    '''
    Validate the name structure of a Sentinel-2 tile. This tests whether the input tile format is correct.
    
    Args:
        tile: A string containing the name of the tile to to download.
    '''
    
    # Tests whether string is in format ##XXX
    name_test = re.match("[0-9]{2}[A-Z]{3}$",tile)
    
    return bool(name_test)
    

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
    assert _validateTile(tile), "The tile name input (%s) does not match the format ##XXX (e.g. 36KWA)."%tile
    
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
    
    print('Found %s matching images'%str(len(products_df)))

    # Where no results for tile    
    if len(products_df) == 0: return products_df
    
    products_df['filesize_mb'] = _get_filesize(products_df)
    
    products_df = products_df[products_df['filesize_mb'] >= float(minsize)]
        
    return products_df


def download(products_df, output_dir = os.getcwd()):
    ''' download(products_df, output_dir = os.getcwd())
    
    Downloads all images from a dataframe produced by sentinelsat.
    
    Args:
        products_df: Pandas dataframe from search() function.
        output_dir: Optionally specify an output directory. Defaults to the present working directory.
    '''
    
    assert os.path.isdir(output_dir), "Output directory doesn't exist."
    
    if products_df.empty is True:
        print('WARNING: No products found to download. Check your search terms.')
        raise
        
    else:
        
        downloaded_files = []
        
        for uuid, filename in zip(products_df['uuid'], products_df['filename']):
            
            if os.path.exists('%s/%s'%(output_dir, filename)):
                print('Skipping file %s, as it has already been downloaded in the directory %s. If you want to re-download it, delete it and run again.'%(filename, output_dir))
            
            elif os.path.exists('%s/%s'%(output_dir, filename[:-5] + '.zip')):
                print('Skipping file %s, as it has already been downloaded and extracted in the directory %s. If you want to re-download it, delete it and run again.'%(filename, output_dir))
            
            elif os.path.exists('%s/%s'%(output_dir, filename)):
                print('Skipping file %s, as it has already been downloaded and extracted in the directory %s. If you want to re-download it, delete it and run again.'%(filename, output_dir))

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
    

def main(username, password, tiles, level = '1C', start = '20150523', end = datetime.datetime.today().strftime('%Y%m%d'), maxcloud = 100, minsize = 25., output_dir = os.getcwd(), remove = False):
    """main(username, password, tiles, level = '1C', start = '20150523', end = datetime.datetime.today().strftime('%Y%m%d'), maxcloud = 100, minsize = 25., output_dir = os.getcwd(), remove = False)
    
    Download Sentinel-2 data from the Copernicus Open Access Hub, specifying a particular tile, date ranges and degrees of cloud cover. This is the function that is initiated from the command line.
    
    Args:
        username: Scihub username. Sign up at https://scihub.copernicus.eu/.
        password: Scihub password.
        tiles: A string containing the name of the tile to to download, or a list of tiles.
        level: Download level '1C' (default) or '2A' data.
        start: Start date for search in format YYYYMMDD. Defaults to '20150523'.
        end: End date for search in format YYYYMMDD. Defaults to today's date.
        maxcloud: An integer of maximum percentage of cloud cover to download. Defaults to 100 %% (download all images, regardless of cloud cover).
        minsize: A float with the minimum filesize to download in MB. Defaults to 25 MB.  Be aware, file sizes smaller than this can result sen2three crashing.
        output_dir: Optionally specify an output directory. Defaults to the present working directory.
        remove: Boolean value, which when set to True deletes level 1C .zip files after decompression is complete. Defaults to False.
    """
    
    # Allow download of single tile
    if type(tiles) == str: tiles = [tiles]
    
    for tile in tiles:
                
        # Connect to API (or reconnect, after timeout)
        connectToAPI(username, password)
    
        # Search for files, return a data frame containing details of matching Sentinel-2 images
        products = search(tile, level = level, start = start, end = end, maxcloud = maxcloud, minsize = minsize)
        
        # Where no data
        if len(products) == 0: continue
        
        # Download products
        zip_files = download(products, output_dir = output_dir)
        
        # Decompress data
        decompress(zip_files, output_dir = output_dir, remove = remove)
        


if __name__ == '__main__':

    # Set up command line parser
    parser = argparse.ArgumentParser(description = 'Download Sentinel-2 data from the Copernicus Open Access Hub, specifying a particular tile, date ranges and degrees of cloud cover.')

    parser._action_groups.pop()
    required = parser.add_argument_group('Required arguments')
    optional = parser.add_argument_group('Optional arguments')

    # Required arguments
    required.add_argument('-u', '--user', type = str, required = True, help = "Scihub username")
    required.add_argument('-p', '--password', type = str, metavar = 'PASS', required = True, help = "Scihub password")
    required.add_argument('-t', '--tiles', type = str, required = True, nargs = '*', help = "Sentinel 2 tile name, in format ##XXX")
    
    # Optional arguments
    optional.add_argument('-l', '--level', type = str, default = '1C', help = "Set to search and download level '1C' (default) or '2A' data. Note that L2A data may not be available at all locations.")
    optional.add_argument('-s', '--start', type = str, default = '20150523', help = "Start date for search in format YYYYMMDD. Defaults to 20150523.")
    optional.add_argument('-e', '--end', type = str, default = datetime.datetime.today().strftime('%Y%m%d'), help = "End date for search in format YYYYMMDD. Defaults to today's date.")
    optional.add_argument('-c', '--cloud', type = int, default = 100, metavar = '%', help = "Maximum percentage of cloud cover to download. Defaults to 100 %% (download all images, regardless of cloud cover).")
    optional.add_argument('-m', '--minsize', type = int, default = 25., metavar = 'MB', help = "Minimum file size to download in MB. Defaults to 25 MB.")
    optional.add_argument('-o', '--output_dir', type = str, metavar = 'PATH', default = os.getcwd(), help = "Specify an output directory. Defaults to the present working directory.")
    optional.add_argument('-r', '--remove', action='store_true', default = False, help = "Remove level 1C .zip files after decompression.")
        
    # Get arguments from command line
    args = parser.parse_args()
    
    # Run through entire processing sequence
    main(args.user, args.password, args.tiles, level = args.level, start = args.start, end = args.end, maxcloud = args.cloud, minsize = args.minsize, output_dir = args.output_dir, remove = args.remove)
