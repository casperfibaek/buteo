import argparse
import datetime
import glob
import numpy as np
import os
import pandas
import time
import zipfile
import sentinelsat
import pdb


# Let API be accessed by other functions


#################################################
### Functions for downloading Sentinel-2 data ###
#################################################

def connectToAPI(username, password):
    '''
    Connect to the SciHub API with sentinelsat. Sign up at https://scihub.copernicus.eu/.
    
    Args:
        username: Scihub username. 
        password: Scihub password.        
    '''
      
    # Connect to Sentinel API
    return  sentinelsat.SentinelAPI(username, password, 'https://scihub.copernicus.eu/dhus')
    

def _buildWkt(search_area):
    """
    Function to build a well-known text polygon from a list of extents.
    
    Args:
        search_area: A list in the format [minlon, minlat, maxlon, maxlat]
    
    Returns:
        A wkt POLYGON string.
    """
    
    # Get strings from search_area floats
    lonmin, latmin, lonmax, latmax = [str(float(i)) for i in search_area]
    
    wkt = 'POLYGON((%s %s,%s %s,%s %s,%s %s,%s %s))'%(lonmin,latmin,lonmax,latmin,lonmax,latmax,lonmin,latmax,lonmin,latmin)
    
    return wkt


def search(search_area, api_connection, start = '20140403', producttype='GRD', end = datetime.datetime.today().strftime('%Y%m%d'), direction= '*'):
    """search(search_area, start = '20140403', end = datetime.datetime.today().strftime('%Y%m%d'), direction= '*')
    
    Searches for Sentinel-1 GRD IW images that meet conditions of date range and extent.
    
    Args:
        search_area: A list in the format [minlon, minlat, maxlon, maxlat]
        start: Start date for search in format YYYYMMDD. Start date may not precede 20140403, the launch date of Sentinel1-. Defaults to 20140403.
        end: End date for search in format YYYYMMDD. Defaults to today's date.
    
    Returns:
        A pandas dataframe with details of scenes matching conditions.
    """
    
    # Set up start and end dates
    startdate = sentinelsat.format_query_date(start)
    enddate = sentinelsat.format_query_date(end)
    
    # Build a POLYGON wkt
    search_polygon = _buildWkt(search_area)
    
    # Search data, filtering by options.
    products = api_connection.query(search_polygon, beginposition = (startdate,enddate), platformname = 'Sentinel-1', producttype = producttype, orbitdirection = direction, sensoroperationalmode = 'IW')

    # convert to Pandas DataFrame, which can be searched modified before commiting to download()
    products_df = api_connection.to_dataframe(products)
    
    if products_df.empty: raise IOError("No products found. Check your search terms.")
    
    return products_df


def removeDuplicates(products_df, data_dir = os.getcwd()):
    '''removeDuplicates(products_df, data_dir = os.getcwd())
    
    Remove images from search results that have already been downloaded
    
    Args:
        products_df: Pandas dataframe from search() function.
        data_dir: Directory containing Sentinel-1 data. Defaults to current working directory.
    
    Returns:
        A dataframe with duplicate files removed.
    '''
    
    # Add trailing '/' to data directory
    data_dir = data_dir.rstrip('/') + '/'
    
    # Build a list of files that are already downloaded
    drop_files = []
    for filename in products_df['filename']:
        if len(glob.glob(data_dir + filename.split('.')[0] + '*')) > 0:
            drop_files.append(filename)
    
    # And drop them from the pandas dataframe
    products_df = products_df[~products_df['filename'].isin(drop_files)]
    
    return products_df


def _removeZip(zip_file):
    """
    Deletes Level 1C .zip file from disk.
    
    Args:
        zip_file: A Sentinel-2 level 1C .zip file from Copernicus Open Access Data Hub.
    """
    
    assert '_MSIL1C_' in zip_file, "removeZip function should only be used to delete Sentinel-2 level 1C compressed .SAFE files"
    assert zip_file.split('/')[-1][-4:] == '.zip', "removeL1C function should only be used to delete Sentinel-2 level 1C compressed .SAFE files"
    
    os.remove(zip_file)



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


def download(products_df, api_connection, output_dir = os.getcwd()):
    ''' download(products_df, output_dir = os.getcwd())
    
    Downloads all images from a dataframe produced by sentinelsat.
    
    Args:
        products_df: Pandas dataframe from search() function.
        output_dir: Optionally specify an output directory. Defaults to the present working directory.
    '''
    
    assert os.path.isdir(output_dir), "Output directory doesn't exist."
    
    if products_df.empty == True:
        raise IOError("No products found. Check your search terms.")
        
    else:
        
        print('Downloading %s product(s)'%str(len(products_df)))
        # Download selected products
        api_connection.download_all(products_df['uuid'], output_dir)

        
if __name__ == '__main__':
    '''
    '''
        
    print('The sen1mosaic command line interface has been moved! Please use scripts in .../sen1mosaic/cli/ to operate sen2mosaic from the command line.')
    