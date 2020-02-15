
import datetime
import glob
import numpy as np
import os
from osgeo import gdal, osr

import sen1mosaic.IO
import sen1mosaic.preprocess
import sen2mosaic.core


#########################################
### Class for loading Sentinel-1 data ###
#########################################

class LoadScene(object):
    '''
    Load a Sentinel-1 (pre-processed) scene
    '''
        
    def __init__(self, filename):
        '''
        Args:
            filename: The path to a Sentinel-1 .dim file
        '''
                
        # Format filename, and check that it exists
        self.filename = self.__checkFilename(filename)
                
        # Get file format
        self.file_format = self.__getFormat()
        
        # Save satellite name
        self.satellite = 'S1'
        
        # Save image type (S1_single, S1_dual, S2)
        self.image_type = self.__getImageType()
        
        self.tile = self.__getTileID()
        
        self.__getMetadata()
        
        # Define source metadata
        self.metadata = sen2mosaic.core.Metadata(self.extent, self.resolution, self.EPSG)
        
        
    def __checkFilename(self, filename):
        '''
        Test that the file exists
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
        
        if self.filename.split('/')[-1].split('.')[-1] == 'dim':
            return 'BEAM-DIMAP'

        else:
            print('File %s does not match any expected file pattern'%self.filename)
            raise IOError
    
    def __getImageType(self):
        '''
        Test whethere S1 file is single ('S1single') or dual ('S1dual') polarised 
        '''
        
        if len(glob.glob(self.filename.split('.dim')[0] +'.data/*0_VH*.img')) > 0:
            image_type = 'S1dual'
        else:
            image_type = 'S1single'
        
        return image_type
    
    def __getTileID(self):
        '''
        '''
        
        return '_'.join(self.filename.split('/')[-1].split('_')[-4:])
    
    def __getMetadata(self):
        '''
        Extract metadata from the Sentinel-1 file.
        '''
        
        self.extent, self.EPSG, self.resolution, self.datetime, self.overpass = sen1mosaic.IO.loadMetadata(self.filename)
    
    def __getImagePath(self, pol = 'VV'):
        '''
        Get the path to a mask or polarisation
        '''

        # Identify source file following the standardised file pattern
        
        image_path = glob.glob(self.filename.split('.dim')[0]+'.data/*0_%s*.img'%pol)
        
        if len(image_path) == 0:
            raise IOError
        
        return image_path[0]
        
    def testInsideTile(self, md_dest):
        '''
        Function that uses metadata class to test whether any part of a source data falls inside destination tile.
        
        Args:
            md_dest: Metadata class from utilities.Metadata() representing the destination image extent.
            
        Returns:
            A boolean (True/False) value.
        '''
        
        # Set up function to translate coordinates from source to destination
        tx = osr.CoordinateTransformation(self.metadata.proj, md_dest.proj)
        
        # And translate the source coordinates
        ulx, uly, z = tx.TransformPoint(self.metadata.ulx, self.metadata.uly)
        lrx, lry, z = tx.TransformPoint(self.metadata.lrx, self.metadata.lry) 
        
        # Determine whether image is outside of tile
        out_of_tile =  ulx >= md_dest.lrx or \
                       lrx <= md_dest.ulx or \
                       uly <= md_dest.lry or \
                       lry >= md_dest.uly
        
        return out_of_tile == False
    
    def testInsideDate(self, start = '20140101', end = datetime.datetime.today().strftime('%Y%m%d')):
        '''
        Function that uses metadata class to test whether a tile falls within the specified time range.
        
        Args:
            start: Start date to process, in format 'YYYYMMDD' Defaults to start of Sentinel-2 era.
            end: End date to process, in format 'YYYYMMDD' Defaults to today's date.
            
        Returns:
            A boolean (True/False) value.
        '''
                
        start = datetime.datetime.strptime(start,'%Y%m%d')
        end = datetime.datetime.strptime(end,'%Y%m%d')
        
        if self.datetime > end:
            return False
        if self.datetime < start:
            return False
        
        return True
    
    def testPolsarisation(self, pol):
        '''
        Function to test whether polarisation is available in the tile.
        
        Args:
            scene: Object from utilties.LoadScene()
            pol: 'VV' or 'VH'
        '''
        
        if self.image_type == 'S1dual':
            return True
        if self.image_type == 'S1single' and pol == 'VV':
            return True
        
        return False
        
    def getMask(self, pol, md = None):
        '''
        Load the mask to a numpy array.
        '''
        
        assert pol in ['VV', 'VH'], "Polarisation can only be 'VV' or 'VH'."
        if self.image_type == 'S1single':
            assert pol != 'VH', "The Sentinel-1 image must be dual polarised to load the VH polarisation."
            
        # Load mask
        image_path = self.__getImagePath(pol = pol)
        
        data = gdal.Open(image_path, 0).ReadAsArray()
        
        # Keep track of pixels where data are contained
        mask = data != 0
        
        # Reproject?
        if md is not None:
            data = sen2mosaic.IO.reprojectBand(self, mask, md, dtype = 1, resampling = gdal.GRA_Mode).astype(np.bool)
        
        # Return pixels without data
        return mask == 0
    
    def getBand(self, pol, md = None):
        '''
        Load a single polarisation
        
        Args:
            pol: 'VV' or 'VH'
        '''
        
        assert pol in ['VV', 'VH'], "Polarisation can only be 'VV' or 'VH'."
        if self.image_type == 'S1single':
            assert pol != 'VH', "The Sentinel-1 image must be dual polarised to load the VH polarisation."
        
        image_path = self.__getImagePath(pol = pol)
        
        # Load the image
        data = gdal.Open(image_path, 0).ReadAsArray()

        # Reproject?
        if md is not None:
            data = sen2mosaic.IO.reprojectBand(self, data, md, dtype = 6, resampling = gdal.GRA_Average) 
        
        return data

