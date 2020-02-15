
import copy
import datetime
import glob
import numpy as np
import os
from osgeo import gdal, osr
import scipy.ndimage
import skimage.measure
import subprocess
import tempfile

import sen2mosaic.IO
import sen2mosaic.preprocess

import pdb

##################################################
### Class containing geospatial image metadata ###
##################################################

class Metadata(object):
    '''
    This is a generic metadata class for Geospatial data
    '''
    
    def __init__(self, extent, res, EPSG):
        '''
        Args:
            extent: A list in the form [xmin, ymin, xmax, ymax]
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
        self.extent = self.__getExtent(extent)
        
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
        
    def __getExtent(self, extent):
        '''
        '''
        
        assert len(extent) == 4, "Extent must be specified in the format [xmin, ymin, xmax, ymax]"
        assert extent[0] < extent[2], "Extent incorrectly specified: xmin must be lower than xmax."
        assert extent[1] < extent[3], "Extent incorrectly specified: ymin must be lower than ymax."
        
        return extent
        
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



#########################################
### Class for loading Sentinel-2 data ###
#########################################

class LoadScene(object):
    '''
    Load a Sentinel-2 L1C or L2A scene, including metadata
    '''
        
    def __init__(self, filename, resolution = 20):
        '''
        Args:
            granule: The path to a Sentinel-2 granule file
            resolution: The resolution to be loaded (10, 20, or 60 metres).
        '''
        
        # Format granule, and check that it exists
        self.granule = self.__getGranule(filename)
        
        # Format filename
        self.filename = self.__getFilename(filename)
          
        # Get file format info
        self.__getFormat(filename)
        
        # Save satellite name
        self.satellite = 'S2'
        
        # Save image type (deprecated)
        self.image_type = 'S2'
        
        self.resolution = self.__getResolution(resolution)
        
        self.__getMetadata()
        
        # Define source metadata
        self.metadata = Metadata(self.extent, self.resolution, self.EPSG)
        
        # Test that all expected images are present
        self.__checkFilesPresent()
        
        
    def __getGranule(self, granule):
        '''
        Test that the granule exists.
        '''
        
        # Test that file exists
        assert os.path.exists(granule),"Cannot find file %s "%granule
        
        # Get rid of trailing '/' if present
        granule = granule.rstrip()
        
        # Return absolute path
        #granule = os.path.abspath(os.path.realpath(granule))
        granule = os.path.abspath(granule)
        
        # If new file format and user inputs a .SAFE file, forgive and update to granule. Else throw error.
        
        if granule.split('/')[-1][-5:] == '.SAFE':
            
            granule_files = glob.glob('%s/GRANULE/*'%granule)
            
            if len(granule_files) == 1:
                
                granule = granule_files[0]
                
            else:
                
                raise IOError("LoadScene() objects require a single 'granule' file as input (e.g. '...SAFE/GRANULE/L2A_T36KWA_A010242_20170608T080546'). Where a .SAFE file is input with the old Sentinel-2 data format (pre 6th December 2016) and a single '.SAFE' file is input, it's ambiguous which granule should be loaded. Please instead specify a single granule.")
        
        assert granule.split('/')[-3].split('.')[-1] == 'SAFE', 'File %s does not match any expected file pattern. Please input a path to a .SAFE granule (*.SAFE/GRANULE/*).'%granule
        
        return granule

    def __getFilename(self, filename):
        '''
        Format for filename.
        '''
        
        # Get granule name
        granule = self.__getGranule(filename)
        
        # Shorten to filename (ending in .SAFE)
        filename = '/'.join(granule.split('/')[:-2])
        
        assert filename.endswith('.SAFE'), "Filename must end with .SAFE. Input format not recognised. Granule name was %s."%filename
        
        return filename
    
    def __getFormat(self, filename):
        '''
        Get format info for tile 
        '''
        
        filename = self.__getFilename(filename)
        
        # Get format of .SAFE file, which are available in multiple versions.
        try:
            self.level, self.spacecraft_name, self.product_format, self.processing_baseline = sen2mosaic.IO.loadFormat(filename)
            
        except:
            print('Failed to load file format metadata.')
            raise
        
        return
    
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
            self.extent, self.EPSG, self.datetime, self.tile, self.nodata_percent = sen2mosaic.IO.loadMetadata(self.granule, self.resolution, level = self.level)
        except Exception as e:
            print('Failed to load granule metadata: %s'%str(e))
            raise
        
        return
    
    def __getImagePath(self, band, resolution = 20):
        '''
        Get the path to a mask or band (Jpeg2000 format).
        '''

        # Identify source file following the standardised file pattern
        
        if self.level == '2A':
                        
            image_path = glob.glob(self.granule + '/IMG_DATA/R%sm/*%s*%sm.jp2'%(str(resolution), band, str(resolution)))
            
            # In old files the mask can be in the base folder
            if len(image_path) == 0 and band == 'SCL':
                image_path = glob.glob(self.granule + '/IMG_DATA/*%s*%sm.jp2'%(band, str(resolution)))
        
        elif self.level == '1C':
            
            image_path = glob.glob(self.granule + '/IMG_DATA/%s_*_%s.jp2'%(str(self.tile), band))        
                
        assert len(image_path) > 0, "No file found for band: %s, resolution: %s in file %s."%(band, str(resolution), self.granule)
        
        return image_path[0]
    

    def __findGML(self, variety, band = 'B02'):
        '''
        '''
        
        assert variety in ['CLOUDS', 'DEFECT', 'DETFOO', 'NODATA', 'SATURA', 'TECQUA'], 'Variety of L1C mask (%s) not recognised'%str(variety)
        assert band in ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'], 'Band (%s) not recognised'%str(band)
        assert self.level == '1C', "GML cloud masks are only used in Level 1C data."
        
        if variety == 'CLOUDS':
            gml_path = glob.glob(self.granule + '/QI_DATA/MSK_%s_B00.gml'%variety)
        else:
            # Assume all bands approx the same
            gml_path = glob.glob(self.granule + '/QI_DATA/MSK_%s_%s.gml'%(variety, band))
        
        assert len(gml_path) > 0, "No GML file found for file %s"%self.granule
        
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
        
        if self.level == '2A':
            if self.resolution == 10:
                bands.extend(['B08'])
            else:
                bands.extend(['B05', 'B06', 'B07', 'B8A', 'B11', 'B12'])
        
            if self.resolution == 60:
                bands.extend(['B01', 'B09'])
            
            bands.extend(['SCL'])
        
        if self.level == '1C':
            bands.extend(['B01', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'])
                    
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
    
    
    def __improveMask(self, mask, cloud_buffer = 180):
        """
        Function that applied tweaks to the cloud mask output from sen2cor (L2A data) or bundled cloud masks (L1C data). Processes are:
            (1) Changing 'dark features' to 'cloud shadows
            (2) Dilating 'cloud shadows', 'medium probability cloud' and 'high probability cloud'.
            (3) Eroding outer 3 km of the tile.
        
        Args:
            mask: A mask from sen2cor
            cloud_buffer: Buffer to place around clouds, in metres
        
        Returns:
            A numpy array of the SCL mask with modifications.
        """
        
        # Make a copy of the original classification mask
        mask_orig = mask.copy()
        
        # Change cloud shadow to dark areas
        mask[mask == 3] = 2
        
        # Change cloud shadows not within 1800 m of a cloud pixel to dark pixels
        iterations = int(round(1800/float(self.resolution)))
        
        # Identify pixels proximal to any measure of cloud cover
        cloud_dilated = scipy.ndimage.morphology.binary_dilation((np.logical_or(mask==8, mask==9)).astype(np.int), iterations = iterations)
        
        # Set these to dark features
        mask[np.logical_and(np.logical_or(mask == 2, mask == 3), cloud_dilated)] = 3
            
        if cloud_buffer > 0:
            
            # Dilate cloud shadows, med clouds and high clouds by cloud_buffer metres.
            iterations = int(round(float(cloud_buffer) / float(self.resolution), 0))
            
            # Make a temporary dataset to prevent dilated masks overwriting each other
            mask_temp = mask.copy()
                
            for i in [3,8,9]:
                            
                # Grow the area of each input class
                mask_dilate = scipy.ndimage.morphology.binary_dilation((mask==i).astype(np.int), iterations = iterations)
                
                # Set dilated area to the same value as input class (except for high probability cloud, set to medium)
                mask_temp[mask_dilate] = i if i is not 9 else 8
            
            mask = mask_temp.copy()
        
        # Erode outer 0.6 km of image tile (should retain overlap)
        iterations = int(round(600 / float(self.resolution)))
        
        # Grow the area of nodata pixels (everything that is equal to 0)
        mask_erode = scipy.ndimage.morphology.binary_dilation((mask_orig == 0).astype(np.int), iterations=iterations)
        
        # Set these eroded areas to 0
        mask[mask_erode == True] = 0
                    
        return mask
    
    def processToL2A(self, gipp = None, output_dir = os.getcwd(), resolution = 0, sen2cor = 'L2A_Process', sen2cor_255 = None, verbose = False):
        '''
        Function to process L1C data to L2A using sen2cor.
        
        Args:
            gipp: Optionally specify a copy of the L2A_GIPP.xml file in order to tweak options.
            output_dir: Optionally specify an output directory. Defaults to current working directory.
            resolution: Optionally specify a resolution (10, 20 or 60) meters. Defaults to 0, which processes all three.
            sen2cor: Path to sen2cor (v2.8) (defaults to 'L2A_Process')
            sen2cor_255: Path to sen2cor_255 (v2.5.5), required if processing an old format of Sentinel-2 data.
        Returns:
            Absolute file path to L2A output file.
        '''
        
        assert self.level == '1C', "Only level 1C data can be processed to L2A."
        
        # Process L1C to L2A
        outfile = sen2mosaic.preprocess.processToL2A(self.granule, gipp = gipp, output_dir = output_dir, resolution = resolution, sen2cor = sen2cor, sen2cor_255 = sen2cor_255, product_format = self.product_format, verbose = verbose)
                
        return outfile
    
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
    
    def testInsideDate(self, start = '20150101', end = datetime.datetime.today().strftime('%Y%m%d')):
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
    
        
    def getMask(self, improve = False, md = None, chunk = None, cloud_buffer = 180, temp_dir = '/tmp'):
        '''
        Load the mask to a numpy array.
        
        Args:
            improve: Set to True to apply improvements to the Sentinel-2 mask (recommended)
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
                                    
                # Load mask into memory
                mask = gdal.Open(image_path, 0).ReadAsArray(*chunk)
            
            # Expand 20 m resolution mask to match 10 metre image resolution if required
            if self.metadata.res == 10:
                mask = scipy.ndimage.zoom(mask, 2, order = 0)
        
        # Enhance mask?
        if improve and mask.sum() > 0:
            mask = self.__improveMask(mask, cloud_buffer = cloud_buffer)

        # Reproject?        
        if md is not None:
             mask = sen2mosaic.IO.reprojectBand(self, mask, md, dtype = 1)
         
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
        if chunk is not None: chunk = np.round(np.array(chunk) / float(zoom),0).astype(np.int).tolist()
        
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
            data = sen2mosaic.IO.reprojectBand(self, data, md, dtype = 2)
        
        return data
