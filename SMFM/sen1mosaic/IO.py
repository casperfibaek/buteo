
import datetime
import glob
import numpy as np
import os
from osgeo import gdal, osr
import xml.etree.ElementTree as ET

import pdb
import sen1mosaic.core

###########################
### Sentinel-1 metadata ###
###########################

def loadMetadata(dim_file):
    '''
    Function to extract georefence info from Sentinel-1 GRD data.
    
    Args:
        dim_file: 
        
    Returns:
        A list describing the extent of the .dim file, in the format [xmin, ymin, xmax, ymax].
        EPSG code of the coordinate reference system of the image
        The image resolution
    '''
        
    assert os.path.exists(dim_file), "The location %s does not contain a Sentinel-1 .dim file."%dim_file
    
    tree = ET.ElementTree(file = dim_file)
    root = tree.getroot()
    
    # Get array size
    size = root.find("Raster_Dimensions")  
    nrows = int(size.find('NROWS').text)
    ncols = int(size.find('NCOLS').text)
    
    geopos = root.find("Geoposition/IMAGE_TO_MODEL_TRANSFORM").text.split(',')
    ulx = float(geopos[4])
    uly = float(geopos[5])
    xres = float(geopos[0])
    yres = float(geopos[3])
    lrx = ulx + (xres * ncols)
    lry = uly + (yres * nrows)
    extent = [ulx, lry, lrx, uly]
    
    res = abs(xres)
        
    wkt = root.find("Coordinate_Reference_System/WKT").text
    
    srs = osr.SpatialReference(wkt = wkt)
    srs.AutoIdentifyEPSG()
    EPSG = int(srs.GetAttrValue("AUTHORITY", 1))
    
    # Extract date string from filename
    datestring = root.find("Production/PRODUCT_SCENE_RASTER_START_TIME").text.split('.')[0]
    this_datetime = datetime.datetime.strptime(datestring, '%d-%b-%Y %H:%M:%S')
    
    # Get ascending/descending overpass
    overpass = root.find("Dataset_Sources/MDElem/MDElem/MDATTR[@name='PASS']").text
    
    return extent, EPSG, res, this_datetime, overpass


##############################
### Sentinel-1 input files ###
##############################

def prepInfiles(infiles, image_type = 'post'):
    """
    Function to identify valid input files for processing chain
    
    Args:
        infiles: A list of input files, directories, or tiles for Sentinel-1 inputs.
    Returns:
        A list of all Sentinel-1 IW GRD files in infiles.
    """
    
    assert image_type in ['pre', 'post'], "image_type must be 'pre' or 'post'."
    
    # Get absolute path, stripped of symbolic links
    infiles = [os.path.abspath(os.path.realpath(infile)) for infile in infiles]
    
    # List to collate 
    infiles_reduced = []
    
    for infile in infiles:
        
        # If image in the pre-processed state
        if image_type == 'pre':
            
            # Where infile is a directory :
            infiles_reduced.extend(glob.glob('%s/S1?_IW_GRDH_*_????.zip'%infile))
            infiles_reduced.extend(glob.glob('%s/S1?_IW_GRDH_*_????/manifest.safe'%infile))
            infiles_reduced.extend(glob.glob('%s/S1?_IW_GRDH_*_????.SAFE/manifest.safe'%infile))
            
            # Where infile is an unzipped SAFE file
            infiles_reduced.extend(glob.glob('%s/manifest.safe'%infile))
            
            # Where infile is a manifest.safe file
            if infile.split('/')[-1] == 'manifest.safe': infiles_reduced.extend(glob.glob('%s'%infile))
            
            # Where infile is a .zip file
            if infile.split('/')[-1][-4::] == '.zip': infiles_reduced.extend(glob.glob('%s'%infile))            
        
        # If image has come via preprocess.py
        elif image_type == 'post':
            
            # Where infile is a directory
            infiles_reduced.extend(glob.glob('%s/*_??????_??????_??????_??????.dim'%infile))
            
            # Where infile is a .dim file
            if infile.split('.')[-1] == 'dim': infiles_reduced.extend(glob.glob('%s'%infile))
            
            # Where infile is a .data directory
            if infile.split('.')[-1] == 'data': infiles_reduced.extend(glob.glob('%s'%infile).replace('.data','.dim'))
            
    # Strip repeats
    infiles_reduced = list(set(infiles_reduced))
    
    return infiles_reduced


def loadSceneList(infiles, pol = 'VV', md_dest = None, start = '20140101', end = datetime.datetime.today().strftime('%Y%m%d'), sort = True):
    """
    Function to load a list of infiles or all files in a directory as sen1moisac.LoadScene() objects.
    """

    def _sortScenes(scenes):
        '''
        Function to sort a list of scenes by date.
        
        Args:
            scenes: A list of utilitites.LoadScene() Sentinel-1 objects
        Returns:
            A sorted list of scenes
        '''
        
        scenes_out = []
        
        scenes = np.array(scenes)
        
        dates = np.array([scene.datetime for scene in scenes])
        
        for date in np.unique(dates):
            scenes_out.extend(scenes[dates == date].tolist())
        
        return scenes_out
    
    
    assert pol in ['VV', 'VH'], "pol must be 'VV' or 'VH'."
    
    # Prepare input string, or list of files
    source_files = prepInfiles(infiles, image_type = 'post')
    
    scenes = []
    for source_file in source_files:
        
        try:
            
            # Load scene
            scene = sen1mosaic.core.LoadScene(source_file)
            
            # Skip scene if conditions not met
            if md_dest is not None and scene.testInsideTile(md_dest) == False: continue
            if scene.testInsideDate(start = start, end = end) == False: continue
            if scene.testPolsarisation('VV') == False: continue
            
            scenes.append(scene)
        
        except Exception as e:
            print("WARNING: Error in loading scene %s with error '%s'. Continuing."%(source_file,str(e)))   
    
    # Optionally sort
    if sort is not None: scenes = _sortScenes(scenes)
    
    return scenes
