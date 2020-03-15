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



def _reprojectImage(ds_source, ds_dest, md_source, md_dest, resampling = 0):
    '''
    Reprojects a source image to match the coordinates of a destination GDAL dataset.
    
    Args:
        ds_source: A gdal dataset from sen2mosaic.createGdalDataset() containing data to be repojected.
        ds_dest: A gdal dataset from sen2mosaic.createGdalDataset(), with destination coordinate reference system and extent.
        md_source: Metadata class from sen2mosaic.Metadata() representing the source image.
        md_dest: Metadata class from sen2mosaic.Metadata() representing the destination image.
    
    Returns:
        A GDAL array with resampled data
    '''
    
    from osgeo import gdal
    
    def _copyds(ds):
        '''
        Build a copy of an input ds, where performing fix on nodata values
        '''
        
        proj = osr.SpatialReference(wkt=ds.GetProjection())
        proj.AutoIdentifyEPSG()
        epsg = int(proj.GetAttrValue('AUTHORITY',1))
                
        geo_t = ds.GetGeoTransform()
        ulx = geo_t[0]
        lrx = geo_t[0] + (geo_t[1] * ds.RasterXSize)
        lry = geo_t[3] + (geo_t[5] * ds.RasterYSize)
        uly = geo_t[3]
        
        extent = [ulx, lry, lrx, uly]
                
        md = sen1mosaic.core.Metadata(extent, ds.GetGeoTransform()[1], epsg)
        return createGdalDataset(md, dtype = 1)
    
    proj_source = md_source.proj.ExportToWkt()
    proj_dest = md_dest.proj.ExportToWkt()
    
    # Reproject source into dest project coordinates
    gdal.ReprojectImage(ds_source, ds_dest, proj_source, proj_dest, resampling)
            
    ds_resampled = ds_dest.GetRasterBand(1).ReadAsArray()
    
    """
    # This may be required again, but for now leave this out, memory requirement is unpredictable.
    
    # As GDAL fills in all nodata pixels as zero, re-do transfromation with array of ones and re-allocate zeros to nodata. Only run where a nodata value has been assigned to ds_source.
    if ds_source.GetRasterBand(1).GetNoDataValue() is not None:
        ds_source_mask = _copyds(ds_source)
        ds_dest_mask = _copyds(ds_dest)
        #ds_source_mask.GetRasterBand(1).WriteArray(np.ones_like(ds_source.GetRasterBand(1).ReadAsArray()))
        ds_source_mask.GetRasterBand(1).WriteArray(np.ones((ds_source.RasterYSize, ds_source.RasterXSize), dtype = np.bool))
        gdal.ReprojectImage(ds_source_mask, ds_dest_mask, proj_source, proj_dest, gdal.GRA_NearestNeighbour)
        ds_resampled[ds_dest_mask.GetRasterBand(1).ReadAsArray() == 0] = ds_source.GetRasterBand(1).GetNoDataValue()
    """
    
    return np.squeeze(ds_resampled)




def reprojectBand(scene, data, md_dest, dtype = 2, resampling = 0):
    """
    Funciton to load, correct and reproject a Sentinel-2 array
    
    Args:
        scene: A level-2A scene of class sen2mosaic.LoadScene().
        data: The array to reproject
        md_dest: An object of class sen2mosaic.Metadata() to reproject image to.
    
    Returns:
        A numpy array of resampled mask data
    """
    
    # Write mask array to a gdal dataset
    ds_source = createGdalDataset(scene.metadata, data_out = data, dtype = dtype)
        
    # Create an empty gdal dataset for destination
    ds_dest = createGdalDataset(md_dest, dtype = dtype)
    
    # Reproject source to destination projection and extent
    data_resampled = _reprojectImage(ds_source, ds_dest, scene.metadata, md_dest, resampling = resampling)
    
    return data_resampled


def createGdalDataset(md, data_out = None, filename = '', driver = 'MEM', dtype = 3, RasterCount = 1, nodata = None, options = []):
    '''
    Function to create an empty gdal dataset with georefence info from metadata dictionary.

    Args:
        md: Object from Metadata() class.
        data_out: Optionally specify an array of data to include in the gdal dataset.
        filename: Optionally specify an output filename, if image will be written to disk.
        driver: GDAL driver type (e.g. 'MEM', 'GTiff'). By default this function creates an array in memory, but set driver = 'GTiff' to make a GeoTiff. If writing a file to disk, the argument filename must be specified.
        dtype: Output data type. Default data type is a 16-bit unsigned integer (gdal.GDT_Int16, 3), but this can be specified using GDAL standards.
        options: A list containing other GDAL options (e.g. for compression, use [compress='LZW'].

    Returns:
        A GDAL dataset.
    '''
    from osgeo import gdal, osr
        
    gdal_driver = gdal.GetDriverByName(driver)
    ds = gdal_driver.Create(filename, md.ncols, md.nrows, RasterCount, dtype, options = options)
    
    ds.SetGeoTransform(md.geo_t)
    
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(md.EPSG_code)
    ds.SetProjection(proj.ExportToWkt())
    
    # If a data array specified, add data to the gdal dataset
    if type(data_out).__module__ == np.__name__:
        
        if len(data_out.shape) == 2:
            data_out = np.ma.expand_dims(data_out,2)
        
        for feature in range(RasterCount):
            ds.GetRasterBand(feature + 1).WriteArray(data_out[:,:,feature])
            
            if nodata != None:
                ds.GetRasterBand(feature + 1).SetNoDataValue(nodata)
    
    # If a filename is specified, write the array to disk.
    if filename != '':
        ds = None
    
    return ds
