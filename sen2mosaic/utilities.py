#!/usr/bin/env python
import copy
import cv2
import datetime
import glob
import numpy as np
import os
from osgeo import gdal, osr
import re
import scipy.ndimage
import skimage.measure
import subprocess
import tempfile

import pdb

# Test alternate loading of lxml
import xml.etree.ElementTree as ET

#import lxml.etree as ET


# This module contains functions to help in image mosaicking, masking, preparation and loading. It is used by sen2mosaic, sen1mosaic, and deforest tools.


class Metadata(object):
    '''
    This is a generic metadata class for Geospatial data
    '''
    
    def __init__(self, extent, res, EPSG):
        '''
        Args:
            extent: A list in the form [xmin. ymin, xmax, ymax]
            res: Pixel resolution
            EPSG: The EPSG code of the desired resolution
        '''
           
        
        # Define projection from EPSG code
        self.EPSG_code = EPSG
        
        # Define resolution
        self.res = res
        
        self.xres = float(res)
        self.yres = float(-res)
        
        # Define image extent data
        self.extent = extent
        
        self.ulx = float(extent[0])
        self.lry = float(extent[1])
        self.lrx = float(extent[2])
        self.uly = float(extent[3])
        
        # Get projection
        self.proj = self.__getProjection()
                
        # Calculate array size
        self.nrows = self.__getNRows()
        self.ncols = self.__getNCols()
        
        # Define gdal geotransform (Affine)
        self.geo_t = self.__getGeoT()
        
        
    def __getProjection(self):
        '''
        '''
                
        # Get GDAL projection string
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(self.EPSG_code)
        
        return proj
    
    def __getNRows(self):
        '''
        '''
        
        return int(round((self.lry - self.uly) / self.yres))
    
    def __getNCols(self):
        '''
        '''
        
        return int(round((self.lrx - self.ulx) / self.xres))
    
    def __getGeoT(self):
        '''
        '''
        
        geo_t = (self.ulx, self.xres, 0, self.uly, 0, self.yres)
        
        return geo_t
    
    def createBlankArray(self, dtype = np.uint16):
        '''
        Create a blank array with the extent of the Metadata class.
            
        Args:
            dtype: Data type from numpy, defaults to np.uint16.
            
        Returns:
            A numpy array sized to match the specification of the utilities.Metadata() class.
        '''
        
        return np.zeros((self.nrows, self.ncols), dtype = dtype)
        
        
class LoadScene(object):
    '''
    Load a Sentinel-2, L1C or L2A scene
    '''
        
    def __init__(self, filename, resolution = 20):
        '''
        Args:
            filename: The path to a Sentinel-2 granule file
            resolution: The resolution to be loaded (10, 20, or 60 metres).
        '''
                
        # Format filename, and check that it exists
        self.filename = self.__checkFilename(filename)
                
        # Get file format
        self.file_format = self.__getFormat()
        
        # Save satellite name
        self.satellite = 'S2'
        
        # Save image type (deprecated)
        self.image_type = 'S2'

        self.level = self.__getLevel()
                
        self.resolution = self.__getResolution(resolution)
                   
        self.__getMetadata()
        
        # Define source metadata
        self.metadata = Metadata(self.extent, self.resolution, self.EPSG)
        
        # Test that all expected images are present
        self.__checkFilesPresent()
        
        
    def __checkFilename(self, filename):
        '''
        Test that the granule exists
        '''
        
        # Get rid of trailing '/' if present
        filename = filename.rstrip()
        
        # Test that file exists
        assert os.path.exists(filename),"Cannot find file %s "%filename
        
        return filename
    
    def __getFormat(self):
        '''
        Test that the file of of an appropriate format
        '''
        
        if self.filename.split('/')[-3].split('.')[-1] == 'SAFE':
            file_type = 'SAFE'
        
        assert file_type == 'SAFE', 'File %s does not match any expected file pattern'%self.filename
        
        return file_type
        
    def __getLevel(self):
        '''
        Determines the level of Sentinel-2 image.
        
        Returns:
            An integer
        '''
        
        if self.filename.split('/')[-1][:3] == 'L2A':    
            level = '2A'
        elif self.filename.split('/')[-1][:3] == 'L1C':    
            level = '1C'
        elif self.filename.split('/')[-1].split('_')[3] == 'L2A':
            level = '2A'
        elif self.filename.split('/')[-1].split('_')[3] == 'L1C':
            level = '1C'
        else:
            level = 'unknown'
        
        return level
    
    def __getResolution(self, resolution):
        '''
        Makes sure that the resolution matches a Sentinel-2 resolution
        '''
        
        assert resolution in [10, 20, 60], "Resolution must be 10, 20 or 60 m."
        
        return resolution
    
                
    def __getMetadata(self):
        '''
        Extract metadata from the Sentinel-2 file.
        '''
        
        try:
            self.extent, self.EPSG, self.datetime, self.tile, self.nodata_percent = getS2Metadata(self.filename, self.resolution, level = self.level)
        except Exception as e:
            print(str(e))
            print('Failed to load metadata.')
    
    def __getImagePath(self, band, resolution = 20):
        '''
        Get the path to a mask or band (Jpeg2000 format).
        '''

        # Identify source file following the standardised file pattern
        
        if self.level == '2A':
                        
            image_path = glob.glob(self.filename + '/IMG_DATA/R%sm/*%s*%sm.jp2'%(str(resolution), band, str(resolution)))
            
            # In old files the mask can be in the base folder
            if len(image_path) == 0 and band == 'SCL':
                image_path = glob.glob(self.filename + '/IMG_DATA/*%s*%sm.jp2'%(band, str(resolution)))
        
        elif self.level == '1C':
            
            image_path = glob.glob(self.filename + '/IMG_DATA/%s_*_%s.jp2'%(str(self.tile), band))        
        
        assert len(image_path) > 0, "No file found for band: %s, resolution: %s in file %s."%(band, str(resolution), self.filename)
        
        return image_path[0]
    

    def __findGML(self, variety, band = 'B02'):
        '''
        '''
        
        assert variety in ['CLOUDS', 'DEFECT', 'DETFOO', 'NODATA', 'SATURA', 'TECQUA'], 'Variety of L1C mask (%s) not recognised'%str(variety)
        assert band in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'], 'Band (%s) not recognised'%str(band)
        assert self.level == '1C', "GML cloud masks are only used in Level 1C data."
        
        if variety == 'CLOUDS':
            gml_path = glob.glob(self.filename + '/QI_DATA/MSK_%s_B00.gml'%variety)
        else:
            # Assume all bands approx the same
            gml_path = glob.glob(self.filename + '/QI_DATA/MSK_%s_%s.gml'%(variety, band))
        
        assert len(gml_path) > 0, "No GML file found for file %s"%self.filename
        
        return gml_path[0]
    

    def __loadGML(self, gml_path, chunk = None, temp_dir = '/tmp'):
        '''
        Loads a cloud mask from the Sentinel-2 level 1C data product
        '''
        
        
        # Generate a temporary output file
        try:
            temp_tif = tempfile.mktemp(suffix='.tif', dir=temp_dir)
        except:
            raise IOError("Failed to write temporary mask file. Temp directory (%s) probably lacks write permission. Try setting a different output directory, or changing permissions with chmod."%temp_dir)
        
        # Rasterize to temp file
        cmd = ['gdal_rasterize', '-burn', '1' ,'-of', 'GTiff', '-te', str(self.extent[0]), str(self.extent[1]), str(self.extent[2]), str(self.extent[3]), '-tr', str(self.resolution), str(self.resolution), gml_path, temp_tif]
        
        try:
            # Load vector mask, rasterize with gdal_rasterize, and load into memory
            with open(os.devnull, 'w') as devnull:
                gdal_output = subprocess.check_output(' '.join(cmd), shell = True, stderr=devnull)              
                if chunk is not None:
                    mask = gdal.Open(temp_tif,0).ReadAsArray(chunk[0], chunk[1], chunk[2], chunk[3])
                else:
                    mask = gdal.Open(temp_tif,0).ReadAsArray()
                # Delete temp file
                gdal_removed = subprocess.check_output('rm ' + temp_tif, shell = True, stderr=devnull)
        except:
            # Occasionally the mask GML file is empty. Assume all pixels should be masked in this case
            mask = np.zeros((int((self.metadata.extent[3] - self.metadata.extent[1]) / self.metadata.res), int((self.metadata.extent[2] - self.metadata.extent[0]) / self.metadata.res))) + 1
            # Mimics behaviour of gdal.Open where chunk is larger than image
            if chunk is not None:
                mask = np.zeros((chunk[3], chunk[2])) #Untested
        return mask == 1

    def __checkFilesPresent(self):
        '''
        Test that all expected images are present.
        '''
        
        # Get a list of expected bands
        bands = ['B02', 'B03', 'B04']
        
        if self.resolution == 10:
            bands.extend(['B08'])
        else:
            bands.extend(['B05', 'B06', 'B07', 'B8A', 'B11', 'B12'])
        
        if self.resolution == 60: bands.extend(['B01', 'B09'])
        
        if self.level == '2A': bands.extend(['SCL'])
            
        # Test that each is present as expected
        for band in bands:
            if band == 'SCL' and self.resolution == 10:
                file_path = self.__getImagePath(band, resolution = 20)
            else:
                file_path = self.__getImagePath(band, resolution = self.resolution)
        
        # For case of L1C data        
        if self.level == '1C':
            mask_types = ['CLOUDS', 'DEFECT', 'DETFOO', 'NODATA', 'SATURA', 'TECQUA']
            for mask_type in mask_types:
                for band in bands:
                    file_path = self.__findGML(mask_type, band)
        
        
    def getMask(self, correct = False, md = None, chunk = None, cloud_buffer = 180, temp_dir = '/tmp'):
        '''
        Load the mask to a numpy array.
        
        Args:
            correct: Set to True to apply improvements to the Sentinel-2 mask (recommended)
        '''
        
        if self.level == '1C':
            
            # Rasterize and load GML cloud mask for L1C data
            gml_path = self.__findGML('CLOUDS')
            
            mask_clouds = self.__loadGML(gml_path, chunk = chunk, temp_dir = temp_dir)
             
            # Get area outside of satellite overpass using B02
            mask_nodata = self.getBand('B02', chunk = chunk) == 0
             
            # Initiate mask to pass all (4 = vegetation)
            mask = np.zeros_like(mask_clouds, dtype = np.int) + 4
            mask[mask_clouds] = 9
            mask[mask_nodata] = 0
            
        # Load mask at appropriate resolution
        elif self.level == '2A':
            
            if self.metadata.res in [20, 60]:
                image_path = self.__getImagePath('SCL', resolution = self.resolution)
            else:
                # In case of 10 m image, use 20 m mask
                image_path = self.__getImagePath('SCL', resolution = 20)
            
            # Load the image (.jp2 format)
            if chunk is None:
                
                # Load mask into memory
                mask = gdal.Open(image_path, 0).ReadAsArray()
                
            else:
                if self.resolution == 10:
                    
                    # Correct chunk size for loading 10 m band
                    chunk = [int(round(c / 2.)) for c in chunk]
                    
                    #[int(round(chunk[0]/2.)), int(round(chunk[1]/2.)), int(round(chunk[2]/2.)), int(round(chunk[3]/2.))]
                
                # Load mask into memory
                mask = gdal.Open(image_path, 0).ReadAsArray(*chunk)
            
            # Expand 20 m resolution mask to match 10 metre image resolution if required
            if self.metadata.res == 10:
                mask = scipy.ndimage.zoom(mask, 2, order = 0)
        
        # Enhance mask?
        if correct and mask.sum() > 0:
            mask = improveMask(mask, self.resolution, cloud_buffer = cloud_buffer)

        # Reproject?        
        if md is not None:
             mask = reprojectBand(self, mask, md, dtype = 1)
         
        return mask
    
    
    def getBand(self, band, md = None, chunk = None):
        '''
        Load a Sentinel-2 band to a numpy array.
        '''
        
        bands_10 = ['B02', 'B03', 'B04', 'B08']            
        bands_20 = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
        bands_60 = ['B01', 'B09']
        
        # Default size
        zoom = 1
        
        # Determine the resolution of the chosen band, and how to rescale it to match resolution
        if self.level == '1C':
            image_path = self.__getImagePath(band)
            
            if self.resolution == 10:
                if band in bands_20: zoom = 2
                elif band in bands_60: zoom = 6
            elif self.resolution == 20:
                if band in bands_10: zoom = 0.5
                elif band in bands_60: zoom = 3
            elif self.resolution == 60:
                if band in bands_10: zoom = 1. / 6
                elif band in bands_20: zoom = 1./ 3
        
        # Do the same for level 2A data, but without need for down-scaling which is performed by sen2cor
        if self.level == '2A':
            if self.resolution == 10:
                if band in bands_10:
                    image_path = self.__getImagePath(band, resolution = 10)
                elif band in bands_20:
                    image_path = self.__getImagePath(band, resolution = 20)
                    zoom = 2
                else:
                    image_path = self.__getImagePath(band, resolution = 60)
                    zoom = 6
            elif self.resolution == 20:
                if band in bands_60:
                    image_path = self.__getImagePath(band, resolution = 60)
                    zoom = 3
                else:
                    image_path = self.__getImagePath(band, resolution = 20)
            else:
                image_path = self.__getImagePath(band, resolution = 60)
        
        # Re-cast chunk based on upcoming zoom factor
        chunk = np.round(np.array(chunk) / float(zoom),0).astype(np.int).tolist()
        
        # Load the image (.jp2 format)
        if chunk is None:
            data = gdal.Open(image_path, 0).ReadAsArray()
        else:
            data = gdal.Open(image_path, 0).ReadAsArray(*chunk)
        
        # Expand coarse resolution band to match image resolution if required
        if zoom > 1:
             data = scipy.ndimage.zoom(data, zoom, order = 0)
        if zoom < 1:
            data = np.round(skimage.measure.block_reduce(data, block_size = (int(1./zoom), int(1./zoom)), func = np.mean), 0).astype(np.int)

        # Reproject?
        if md is not None:
            data = reprojectBand(self, data, md, dtype = 2)
        
        return data


def reprojectBand(scene, data, md_dest, dtype = 2, resampling = 0):
    """
    Funciton to load, correct and reproject a Sentinel-2 array
    
    Args:
        scene: A level-2A scene of class utilities.LoadScene().
        data: The array to reproject
        md_dest: An object of class utilities.Metadata() to reproject image to.
    
    Returns:
        A numpy array of resampled mask data
    """
    
    # Write mask array to a gdal dataset
    ds_source = createGdalDataset(scene.metadata, data_out = data, dtype = dtype)
        
    # Create an empty gdal dataset for destination
    ds_dest = createGdalDataset(md_dest, dtype = dtype)
    
    # Reproject source to destination projection and extent
    data_resampled = reprojectImage(ds_source, ds_dest, scene.metadata, md_dest, resampling = resampling)
    
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


def reprojectImage(ds_source, ds_dest, md_source, md_dest, resampling = 0):
    '''
    Reprojects a source image to match the coordinates of a destination GDAL dataset.
    
    Args:
        ds_source: A gdal dataset from utilities.createGdalDataset() containing data to be repojected.
        ds_dest: A gdal dataset from utilities.createGdalDataset(), with destination coordinate reference system and extent.
        md_source: Metadata class from utilities.Metadata() representing the source image.
        md_dest: Metadata class from utilities.Metadata() representing the destination image.
    
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
                
        md = Metadata(extent, ds.GetGeoTransform()[1], epsg)
        return createGdalDataset(md, dtype = 1)
    
    proj_source = md_source.proj.ExportToWkt()
    proj_dest = md_dest.proj.ExportToWkt()
    
    # Reproject source into dest project coordinates
    gdal.ReprojectImage(ds_source, ds_dest, proj_source, proj_dest, resampling)
            
    ds_resampled = ds_dest.GetRasterBand(1).ReadAsArray()
    
    # As GDAL fills in all nodata pixels as zero, re-do transfromation with array of ones and re-allocate zeros to nodata. Only run where a nodata value has been assigned to ds_source.
    if ds_source.GetRasterBand(1).GetNoDataValue() is not None:
        
        ds_source_mask = _copyds(ds_source)
        ds_dest_mask = _copyds(ds_dest)
        ds_source_mask.GetRasterBand(1).WriteArray(np.ones_like(ds_source.GetRasterBand(1).ReadAsArray()))
        gdal.ReprojectImage(ds_source_mask, ds_dest_mask, proj_source, proj_dest, gdal.GRA_NearestNeighbour)
        ds_resampled[ds_dest_mask.GetRasterBand(1).ReadAsArray() == 0] = ds_source.GetRasterBand(1).GetNoDataValue()
    
    return ds_resampled


def validateTile(tile):
    '''
    Validate the name structure of a Sentinel-2 tile. This tests whether the input tile format is correct.
    
    Args:
        tile: A string containing the name of the tile to to download.
    
    Returns:
        A boolean, True if correct, False if not.
    '''
    
    # Tests whether string is in format ##XXX
    name_test = re.match("[0-9]{2}[A-Z]{3}$",tile)
    
    return bool(name_test)


def prepInfiles(infiles, level, tile = ''):
    """
    Function to select input granules from a directory, .SAFE file (with wildcards) or granule, based on processing level and a tile.
    
    Args:
        infiles: A string or list of input .SAFE files, directories, or granules for Sentinel-2 inputs
        level: Set to either '1C' or '2A' to select appropriate granules.
        tile: Optionally filter infiles to return only those matching a particular tile
    Returns:
        A list of all matching Sentinel-2 granules in infiles.
    """
    
    assert level in ['1C', '2A'], "Sentinel-2 processing level must be either '1C' or '2A'."
    assert validateTile(tile) or tile == '', "Tile format not recognised. It should take the format '##XXX' (e.g.' 36KWA')."
    
    # Make interable if only one item
    if not isinstance(infiles, list):
        infiles = [infiles]
    
    # Get absolute path, stripped of symbolic links
    infiles = [os.path.abspath(os.path.realpath(infile)) for infile in infiles]
    
    # In case infiles is a list of files
    if len(infiles) == 1 and os.path.isfile(infiles[0]):
        with open(infiles[0], 'rb') as infile:
            infiles = [row.rstrip() for row in infile]
    
    # List to collate 
    infiles_reduced = []
    
    for infile in infiles:
         
        # Where infile is a directory:
        infiles_reduced.extend(glob.glob('%s/*_MSIL%s_*/GRANULE/*'%(infile, level)))
        
        # Where infile is a .SAFE file
        if '_MSIL%s_'%level in infile.split('/')[-1]: infiles_reduced.extend(glob.glob('%s/GRANULE/*'%infile))
        
        # Where infile is a specific granule 
        if infile.split('/')[-2] == 'GRANULE': infiles_reduced.extend(glob.glob('%s'%infile))
    
    # Strip repeats (in case)
    infiles_reduced = list(set(infiles_reduced))
    
    # Reduce input to infiles that match the tile (where specified)
    infiles_reduced = [infile for infile in infiles_reduced if ('_T%s'%tile in infile.split('/')[-1])]
    
    # Reduce input files to only L1C or L2A files
    infiles_reduced = [infile for infile in infiles_reduced if ('_MSIL%s_'%level in infile.split('/')[-3])]
    
    return infiles_reduced



def getSourceFilesInTile(scenes, md_dest, start = '20150101', end = datetime.datetime.today().strftime('%Y%m%d'), verbose = False):
    '''
    Takes a list of source files as input, and determines where each falls within extent of output tile.
    
    Args:
        scenes: A list of utilitites.LoadScene() Sentinel-2 objects
        md_dest: Metadata class from utilities.Metadata() containing output projection details.
        start: Start date to process, in format 'YYYYMMDD' Defaults to start of Sentinel-2 era.
        end: End date to process, in format 'YYYYMMDD' Defaults to today's date.
        verbose: Set True to print progress.
        
    Returns:
        A reduced list of scenes containing only files that will contribute to each tile.
    '''
    
    def testOutsideTile(md_source, md_dest):
        '''
        Function that uses metadata class to test whether any part of a source data falls inside destination tile.
        
        Args:
            md_source: Metadata class from utilities.Metadata() representing the source image.
            md_dest: Metadata class from utilities.Metadata() representing the destination image.
            
        Returns:
            A boolean (True/False) value.
        '''
        
        import osr
            
        # Set up function to translate coordinates from source to destination
        tx = osr.CoordinateTransformation(md_source.proj, md_dest.proj)
            
        # And translate the source coordinates
        md_source.ulx, md_source.uly, z = tx.TransformPoint(md_source.ulx, md_source.uly)
        md_source.lrx, md_source.lry, z = tx.TransformPoint(md_source.lrx, md_source.lry)   
        
        out_of_tile =  md_source.ulx >= md_dest.lrx or \
                    md_source.lrx <= md_dest.ulx or \
                    md_source.uly <= md_dest.lry or \
                    md_source.lry >= md_dest.uly
        
        return out_of_tile
    
    def testOutsideDate(scene, start = '20150101', end = datetime.datetime.today().strftime('%Y%m%d')):
        '''
        Function that uses LoadScene class to test whether a tile falls within the specified time range.
        
        Args:
            scene: Object from utilities.LoadScene()
            start: Start date to process, in format 'YYYYMMDD' Defaults to start of Sentinel-2 era.
            end: End date to process, in format 'YYYYMMDD' Defaults to today's date.
            
        Returns:
            A boolean (True/False) value.
        '''
                
        start = datetime.datetime.strptime(start,'%Y%m%d')
        end = datetime.datetime.strptime(end,'%Y%m%d')
        
        if scene.datetime > end:
            return True
        if scene.datetime < start:
            return True
        
        return False
        
    # Determine which images are within specified tile bounds
    if verbose: print('Searching for source files within specified tile...')
    
    do_tile = []

    for scene in scenes:
        
        # Skip processing the file if image falls outside of tile area
        if testOutsideTile(scene.metadata, md_dest):
            do_tile.append(False)
            continue
        
        if testOutsideDate(scene, start = start, end = end):
            do_tile.append(False)
            continue
        
        if verbose: print('    Found one: %s'%scene.filename)
        do_tile.append(True)
    
    # Get subset of scenes in specified tile
    scenes_tile = list(np.array(scenes)[np.array(do_tile)])
    
    return scenes_tile


def sortScenes(scenes, by = 'tile'):
    '''
    Function to sort a list of scenes by tile, then by date. This reduces some artefacts in mosaics.
    
    Args:
        scenes: A list of utilitites.LoadScene() Sentinel-2 objects
        by: Set to 'tile' to sort by tile then date, or 'date' to sort by date then tile
    Returns:
        A sorted list of scenes
    '''
    
    scenes_out = []
    
    scenes = np.array(scenes)
    
    dates = np.array([scene.datetime for scene in scenes])
    tiles = np.array([scene.tile for scene in scenes])
    
    if by == 'tile':
        for tile in np.unique(tiles):
            scenes_out.extend(scenes[tiles == tile][np.argsort(dates[tiles == tile])].tolist())
    
    elif by == 'date':
        for date in np.unique(dates):
            scenes_out.extend(scenes[dates == date][np.argsort(tiles[dates == date])].tolist())
    
    return scenes_out



def getS2Metadata(granule_file, resolution = 20, level = '2A', tile = ''):
    '''
    Function to extract georefence info from level 1C/2A Sentinel 2 data in .SAFE format.
    
    Args:
        granule_file: String with /path/to/the/granule folder bundled in a .SAFE file.
        resolution: Integer describing pixel size in m (10, 20, or 60). Defaults to 20 m.

    Returns:
        A list describing the extent of the .SAFE file granule, in the format [xmin, ymin, xmax, ymax].
        EPSG code of the coordinate reference system of the granule
    '''
    
    assert resolution in [10, 20, 60], "Resolution must be 10, 20 or 60 m."
    assert level in ['1C', '2A'], "Product level must be either '1C' or '2A'."
    
    # Remove trailing / from granule directory if present 
    granule_file = granule_file.rstrip('/')
    
    assert len(glob.glob((granule_file + '/*MTD*.xml'))) > 0, "The location %s does not contain a metadata (*MTD*.xml) file."%granule_file
    
    # Find the xml file that contains file metadata
    xml_file = glob.glob(granule_file + '/*MTD*.xml')[0]
    
    # Parse xml file
    tree = ET.ElementTree(file = xml_file)
    root = tree.getroot()
            
    # Define xml namespace
    ns = {'n1':root.tag[1:].split('}')[0]}
    
    # Get array size
    size = root.find("n1:Geometric_Info/Tile_Geocoding[@metadataLevel='Brief']/Size[@resolution='%s']"%str(resolution),ns)
    nrows = int(size.find('NROWS').text)
    ncols = int(size.find('NCOLS').text)
    
    # Get extent data
    geopos = root.find("n1:Geometric_Info/Tile_Geocoding[@metadataLevel='Brief']/Geoposition[@resolution='%s']"%str(resolution),ns)
    ulx = float(geopos.find('ULX').text)
    uly = float(geopos.find('ULY').text)
    xres = float(geopos.find('XDIM').text)
    yres = float(geopos.find('YDIM').text)
    lrx = ulx + (xres * ncols)
    lry = uly + (yres * nrows)
    
    extent = [ulx, lry, lrx, uly]
    
    # Find EPSG code to define projection
    EPSG = root.find("n1:Geometric_Info/Tile_Geocoding[@metadataLevel='Brief']/HORIZONTAL_CS_CODE",ns).text
    EPSG = int(EPSG.split(':')[1])
    
    # Get datetime
    datestring = root.find("n1:General_Info/SENSING_TIME[@metadataLevel='Standard']",ns).text.split('.')[0]
    date = datetime.datetime.strptime(datestring,'%Y-%m-%dT%H:%M:%S')
    
    if level == '2A':
        try:
            # Get nodata percentage based on scene classification
            vegetated = root.find("n1:Quality_Indicators_Info[@metadataLevel='Standard']/L2A_Image_Content_QI/VEGETATION_PERCENTAGE",ns).text
            not_vegetated = root.find("n1:Quality_Indicators_Info[@metadataLevel='Standard']/L2A_Image_Content_QI/NOT_VEGETATED_PERCENTAGE",ns).text
            water = root.find("n1:Quality_Indicators_Info[@metadataLevel='Standard']/L2A_Image_Content_QI/WATER_PERCENTAGE",ns).text
        except:
            # In case of new sen2cor format
            vegetated = root.find("n1:Quality_Indicators_Info[@metadataLevel='Standard']/Image_Content_QI/VEGETATION_PERCENTAGE",ns).text
            not_vegetated = root.find("n1:Quality_Indicators_Info[@metadataLevel='Standard']/Image_Content_QI/NOT_VEGETATED_PERCENTAGE",ns).text
            water = root.find("n1:Quality_Indicators_Info[@metadataLevel='Standard']/Image_Content_QI/WATER_PERCENTAGE",ns).text
            
        nodata_percent = 100. - float(water) - float(vegetated) - float(not_vegetated)
    
    elif level == '1C':
        # Get nodata percentrage based on estimated cloud cover
        cloud_cover = root.find("n1:Quality_Indicators_Info[@metadataLevel='Standard']/Image_Content_QI/CLOUDY_PIXEL_PERCENTAGE", ns).text
        
        nodata_percent = 100. - float(cloud_cover)
    
    if tile == '':
        # Get tile from granule filename
        if granule_file.split('/')[-1].split('_')[1] == 'USER':
            
            # If old file format
            tile = granule_file.split('/')[-1].split('_')[-2]
            
        else:
            
            # If new file format
            tile = granule_file.split('/')[-1].split('_')[1]
    
    return extent, EPSG, date, tile, nodata_percent


def improveMask(data, res, cloud_buffer = 180):
    """
    Function that applied tweaks to the cloud mask output from sen2cor. Processes are: (1) Changing 'dark features' to 'cloud shadows, (2) Dilating 'cloud shadows', 'medium probability cloud' and 'high probability cloud' by 180 m. (3) Eroding outer 3 km of the tile.
    
    Args:
        data: A mask from sen2cor
        res: Integer of resolution to be processed (i.e. 10 m, 20 m, 60 m). This should match the resolution of the mask.
        cloud_buffer: Buffer to place around clouds, in metres
    
    Returns:
        A numpy array of the SCL mask with modifications.
    """
    
    # Make a copy of the original classification mask
    data_orig = data.copy()
    
    # Change cloud shadow to dark areas
    data[data == 3] = 2
    
    # Change cloud shadows not within 1800 m of a cloud pixel to dark pixels
    iterations = int(round(1800/res))
    
    # Identify pixels proximal to any measure of cloud cover
    cloud_dilated = scipy.ndimage.morphology.binary_dilation((np.logical_or(data==8, data==9)).astype(np.int), iterations = iterations)
    
    # Set these to dark features
    data[np.logical_and(np.logical_or(data == 2, data == 3), cloud_dilated)] = 3
    
    #import matplotlib.pyplot as plt
    
    # Don't grow clouds near oceans or large water bodies. First erode away rivers, then grow remainder
    #iterations = int(round(1440/res,0))
    #coastal = scipy.ndimage.morphology.binary_erosion((data==6).astype(np.int), iterations = int(round(60/res,0))) * 1
    #coastal = scipy.ndimage.morphology.binary_dilation((coastal==1).astype(np.int), iterations = iterations)
    
    if cloud_buffer > 0:
        # Dilate cloud shadows, med clouds and high clouds by cloud_buffer metres.
        iterations = int(round(cloud_buffer / res, 0))
        
        # Make a temporary dataset to prevent dilated masks overwriting each other
        data_temp = data.copy()
            
        for i in [3,8,9]:
                        
            # Grow the area of each input class
            mask_dilate = scipy.ndimage.morphology.binary_dilation((data==i).astype(np.int), iterations = iterations)
            
            # Set dilated area to the same value as input class (except for high probability cloud, set to medium)
            data_temp[mask_dilate] = i if i is not 9 else 8
        
        data = data_temp.copy()
    
    # Erode outer 0.6 km of image tile (should retain overlap)
    iterations = int(round(600/res))
    
    # Grow the area of nodata pixels (everything that is equal to 0)
    mask_erode = scipy.ndimage.morphology.binary_dilation((data_orig == 0).astype(np.int), iterations=iterations)
    
    # Set these eroded areas to 0
    data[mask_erode == True] = 0
                
    return data


def histogram_match(source, reference):
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


def colourBalance(image, reference, aggressive = True, verbose = False):
    '''
    Perform colour balancing between a new and reference image. Experimental.
    '''
    
    # Calculate overlap with other images
    overlap = np.logical_and(image.mask == False, reference.mask == False)
    
    # Calculate percent overlap between images
    this_overlap = float(overlap.sum()) / (image.mask == False).sum()
        
    if this_overlap > 0.02 and this_overlap <= 0.5 and aggressive:
        
        if verbose: print('        scaling')
                
        # Gain compensation (simple inter-scene correction)                    
        this_intensity = np.mean(image[overlap])
        ref_intensity = np.mean(reference[overlap])
        
        image[image.mask == False] = np.round(image[image.mask == False] * (ref_intensity/this_intensity),0).astype(np.uint16)
        
    elif this_overlap > 0.5:
        
        if verbose: print('        matching')
        
        image = histogram_match(image, reference)
        
    else:
        
        if verbose: print('        adding')
    
    return image


if __name__ == '__main__':
    '''
    '''
    
    import argparse
    
    # Set up command line parser
    parser = argparse.ArgumentParser(description = "This file contains functions to assist in the mosaicking and masking of Sentinel-2 data. A command line interface for image mosaicking is provided in mosaic.py.")
    
    args = parser.parse_args()
