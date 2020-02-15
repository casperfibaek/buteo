

import datetime
import glob
import numpy as np
import os
from osgeo import gdal, gdalnumeric, osr
from PIL import Image, ImageDraw
import re
import shapefile
import xml.etree.ElementTree as ET

import sen2mosaic

import pdb

### Functions for data input and output, and image reprojection


#########################################
### Geospatial manipulation functions ###
#########################################

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
                
        md = sen2mosaic.core.Metadata(extent, ds.GetGeoTransform()[1], epsg)
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



##########################################
### Generic raster/vector IO functions ###
##########################################

def loadShapefile(shp, md_dest, field = '', field_values = ''):
    """
    Rasterize polygons from a shapefile to match a specified CRS.
        
    Args:
        shp: Path to a shapefile consisting of points, lines and/or polygons. This does not have to be in the same projection as ds.
        md_dest: A metadata file from sen2mosaic.core.Metadata().
        field: Field name to include as part of mask. Defaults to all. If specifying an field, you must also specify an field_value.
        field_values: Field value or list of values (from 'field') to include in the mask. Defaults to all values. If specifying an field_value, you must also specify a 'field'.
        
    Returns:
        A numpy array with a boolean mask delineating locations inside (True) and outside (False) the shapefile given attribute and attribute_value.
    """
    
    if field != '' or field_values != '':
        assert field != '' and field_values != '', "Both  `attribute` and `attribute_value` must be specified."
    
    shp = os.path.expanduser(shp)
    assert os.path.exists(shp), "Shapefile %s does not exist in the file system."%shp
    
    # Allow input of one or more attribute values
    if type(field_values) == str: field_values = [field_values]
    
    def _coordinateTransformer(shp, EPSG_out):
        """
        Generates function to transform coordinates from a source shapefile CRS to EPSG.
        
        Args:
            shp: Path to a shapefile.
        
        Returns:
            A function that transforms shapefile points to EPSG.
        """
                
        driver = ogr.GetDriverByName('ESRI Shapefile')
        ds = driver.Open(shp)
        layer = ds.GetLayer()
        spatialRef = layer.GetSpatialRef()
        
        # Create coordinate transformation
        inSpatialRef = osr.SpatialReference()
        inSpatialRef.ImportFromWkt(spatialRef.ExportToWkt())

        outSpatialRef = osr.SpatialReference()
        outSpatialRef.ImportFromEPSG(EPSG_out)

        coordTransform = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)
        
        return coordTransform


    def _world2Pixel(geo_t, x, y):
        """
        Uses a gdal geomatrix (ds.GetGeoTransform()) to calculate the pixel location of a geospatial coordinate.
        Modified from: http://geospatialpython.com/2011/02/clip-raster-using-shapefile.html.
        
        Args:
            geo_t: A gdal geoMatrix (ds.GetGeoTransform().
            x: x coordinate in map units.
            y: y coordinate in map units.
            buffer_size: Optionally specify a buffer size. This is used when a buffer has been applied to extend all edges of an image, as in rasterizeShapfile().
        
        Returns:
            A tuple with pixel/line locations for each input coordinate.
        """
        ulX = geo_t[0]
        ulY = geo_t[3]
        xDist = geo_t[1]
        yDist = geo_t[5]
        
        pixel = int((x - ulX) / xDist)
        line = int((y - ulY) / yDist)
        
        return (pixel, line)
    
    def _getField(shp, field):
        '''
        Get values from a field in a shapefile attribute table.

        Args:
            shp: A string pointing to a shapefile
            field: A string with the field name of the attribute of interest

        Retuns:
            An array containing all the values of the specified attribute
        '''
        
        assert os.path.isfile(shp), "Shapefile %s does not exist."%shp

        # Read shapefile
        sf = shapefile.Reader(shp)

        # Get the column number of the field of interest
        for n, this_field in enumerate(sf.fields[1:]):

            fieldname = this_field[0]

            if fieldname == field:

                field_n = n

        assert 'field_n' in locals(), "Attribute %s not found in shapefile."%str(field)

        # Extract data type from shapefile. Interprets N (int), F (float) and C (string), sets others to string.
        this_dtype = sf.fields[1:][field_n][1]

        if this_dtype == 'N':
            dtype = np.int
        elif this_dtype == 'F':
            dtype = np.float32
        elif this_dtype == 'C':
            dtype = np.str
        else:
            dtype = np.str

        value_out = []

        # Cycle through records:
        for s in sf.records():
            value_out.append(s[field_n])

        return np.array(value_out, dtype = dtype)
    
    # Create output image
    rasterPoly = Image.new("I", (md_dest.ncols , md_dest.nrows), 0)
    rasterize = ImageDraw.Draw(rasterPoly)
    
    # The shapefile may not have the same CRS as the data, so this will generate a function to reproject points.
    coordTransform = _coordinateTransformer(shp, md_dest.EPSG_code)
    
    # Read shapefile
    sf = shapefile.Reader(shp) 
    
    # Get names of fields
    fields = sf.fields[1:] 
    field_names = [field[0] for field in fields] 
    
    
    # Get shapes
    shapes = np.array(sf.shapes())

    # If extracting a mask for just a single field.
    if field != None:

        shapes = shapes[getField(shp, field) == value]
    
    # For each shape in shapefile...
    # For each shape in shapefile...
    for n, shape in enumerate(shapes):
                
        atr = dict(zip(field_names, r.record))
        
        if attribute_value != '' and atr[attribute] not in attribute_values:
            continue
        
        # Get shape bounding box
        if shape.shapeType == 1 or shape.shapeType == 11:
            # Points don't have a bbox, calculate manually
            sxmin = np.min(np.array(shape.points)[:,0])
            sxmax = np.max(np.array(shape.points)[:,0])
            symin = np.min(np.array(shape.points)[:,1])
            symax = np.max(np.array(shape.points)[:,1])
        else:
            sxmin, symin, sxmax, symax = shape.bbox
            
        
        # Transform points
        sxmin, symin, z = coordTransform.TransformPoint(sxmin, symin)
        sxmax, symax, z = coordTransform.TransformPoint(sxmax, symax)
                
        # Go to the next record if out of bounds
        geo_t = md_dest.geo_t
        if sxmax < geo_t[0]: continue
        if sxmin > geo_t[0] + (geo_t[1] * md_dest.ncols): continue
        if symax < geo_t[3] + (geo_t[5] * md_dest.nrows): continue
        if symin > geo_t[3]: continue
        
        #Separate polygons with list indices
        n_parts = len(shape.parts) #Number of parts
        indices = shape.parts #Get indices of shapefile part starts
        indices.append(len(shape.points)) #Add index of final vertex
        
        for part in range(n_parts):

            if shape.shapeType != 1 and shape.shapeType != 11:

                start_index = shape.parts[part]
                end_index = shape.parts[part+1]
                points = shape.points[start_index:end_index] #Map coordinates

            pixels = [] #Pixel coordinantes

            # Transform coordinates to pixel values
            for p in points:

                # First update points from shapefile projection to ALOS mosaic projection
                lon, lat, z = coordTransform.TransformPoint(p[0], p[1])

                # Then convert map to pixel coordinates using geo transform
                pixels.append(_world2Pixel(tile.geo_t, lon, lat, buffer_size = buffer_size_degrees))

            # Draw the mask for this shape...
            # if a point...
            if shape.shapeType == 0 or shape.shapeType == 1 or shape.shapeType == 11:
                rasterize.point(pixels, n+1)

            # a line...
            elif shape.shapeType == 3 or shape.shapeType == 13:
                rasterize.line(pixels, n+1)

            # or a polygon.
            elif shape.shapeType == 5 or shape.shapeType == 15:
                rasterize.polygon(pixels, n+1)

            else:
                print('Shapefile type %s not recognised!'%(str(shape.shapeType)))

    
    #Converts a Python Imaging Library array to a gdalnumeric image.
    mask = gdalnumeric.fromstring(rasterPoly.tobytes(),dtype=np.uint32)
    mask.shape = rasterPoly.im.size[1], rasterPoly.im.size[0]
        
    return mask


def loadRaster(raster_file, md_dest = None):
    '''
    Load a raster dataset, and optionally reproject
    
    Args:
        raster_file: Path to a GeoTiff or .vrt file.
        md_dest: A metadata file from sen2mosaic.Metadata().
    
    Returns:
        A numpy array
    '''
        
    # Load landcover map
    ds_source = gdal.Open(raster_file, 0)
    
    # If no reprojection required, return array
    if md_dest is None:
        
        return ds_source.GetRasterBand(1)
    
    # Else reproject
    else:
        
        geo_t = ds_source.GetGeoTransform()

        # Get extent and resolution of input raster
        nrows = ds_source.RasterXSize
        ncols = ds_source.RasterYSize
        ulx = float(geo_t[0])
        uly = float(geo_t[3])
        xres = float(geo_t[1])
        yres = float(geo_t[5])
        lrx = ulx + (xres * ncols)
        lry = uly + (yres * nrows)
        extent = [ulx, lry, lrx, uly]
        
        # Get EPSG
        proj = ds_source.GetProjection()
        srs = osr.SpatialReference(wkt = proj)
        srs.AutoIdentifyEPSG()
        EPSG = int(srs.GetAttrValue("AUTHORITY", 1))
        
        # Add source metadata to a dictionary
        md_source = sen2mosaic.Metadata(extent, xres, EPSG)
        
        # Build an empty destination dataset
        ds_dest = createGdalDataset(md_dest, nodata = ds_source.GetRasterBand(1).GetNoDataValue(), dtype = 1)
        
        # And reproject landcover dataset to match input image
        im_rep = np.squeeze(_reprojectImage(ds_source, ds_dest, md_source, md_dest))
        
        return im_rep


###########################
### Sentinel-2 metadata ###
###########################



def loadFormat(filename):
    '''
    Get format information for Sentinel-2 .SAFE file
    
    Args:
        filename: String with /path/to/.SAFE file
    
    Returns:
        Image processing level ('1C'/'2A')
        Spacecraft name ('Sentinel-2A' or 'Sentinel-2B')
        Product format ('SAFE' or 'SAFE_COMPACT'
        Image processing baseline (for sen2cor)
    '''
        
    # Remove trailing / from directory if present
    filename = filename.rstrip('/')
    
    assert len(glob.glob((filename + '/*MTD*.xml'))) > 0, "The location %s does not contain a metadata (*MTD*.xml) file."%filename
    
    # Find the xml file that contains file metadata
    xml_file = glob.glob(filename + '/*MTD*.xml')[0]
    
    # Parse xml file
    tree = ET.ElementTree(file = xml_file)
    root = tree.getroot()

    # Define xml namespace
    ns = {'n1':root.tag[1:].split('}')[0]}
    
    # Determine file structure
    product_info = 'Product_Info'
    if root.find("n1:General_Info/%s"%product_info,ns) is None: product_info = 'L2A_Product_Info'
    
    # Get processing level
    processing_level = root.find("n1:General_Info/%s/PROCESSING_LEVEL"%product_info, ns).text
    
    # Translate level
    level = '2A' if '2A' in processing_level else '1C'
    
    # Get processing baseline
    processing_baseline = root.find("n1:General_Info/%s/PROCESSING_BASELINE"%product_info, ns).text
    
    # Get file format ('SAFE' or 'SAFE_COMPACT')    
    format_pos = root.find("n1:General_Info/%s/Query_Options[@completeSingleTile='true']/PRODUCT_FORMAT"%product_info,ns)
    
    if format_pos is None: format_pos = root.find("n1:General_Info/%s/Query_Options/PRODUCT_FORMAT"%product_info,ns)
    
    product_format = format_pos.text
    
    # Get spacecraft_name ('Sentinel-2A' or 'Sentinel-2B')
    spacecraft_name = root.find("n1:General_Info/%s/Datatake/SPACECRAFT_NAME"%product_info,ns).text
    
    return level, spacecraft_name, product_format, processing_baseline

    
def loadMetadata(granule_file, resolution = 20, level = '2A', tile = ''):
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



##############################
### Sentinel-2 input files ###
##############################

def prepInfiles(infiles, level, tile = ''):
    """
    Function to select input granules from a directory, .SAFE file (with wildcards) or granule, based on processing level and a tile. Used by command line interface to identify input files.
    
    Args:
        infiles: A string or list of input .SAFE files, directories, or granules for Sentinel-2 inputs
        level: Set to either '1C' or '2A' to select appropriate granules.
        tile: Optionally filter infiles to return only those matching a particular tile
    Returns:
        A list of all matching Sentinel-2 granules in infiles.
    """
    
    assert level in ['1C', '2A'], "Sentinel-2 processing level must be either '1C' or '2A'."
    assert bool(re.match("[0-9]{2}[A-Z]{3}$",tile)) or tile == '', "Tile format not recognised. It should take the format '##XXX' (e.g. '36KWA')."
    
    # Make interable if only one item
    if not isinstance(infiles, list):
        infiles = [infiles]
    
    # Get absolute path, stripped of symbolic links
    #infiles = [os.path.abspath(os.path.realpath(infile)) for infile in infiles]
    
    # In case infiles is a list of files
    if len(infiles) == 1 and os.path.isfile(infiles[0]):
        with open(infiles[0], 'rb') as infile:
            infiles = [row.rstrip() for row in infile]
    
    # List to collate 
    infiles_reduced = []
    
    for infile in infiles:
        
        # Remove trailing /, if present
        infile = infile.rstrip('/')
         
        # Where infile is a directory:
        infiles_reduced.extend(glob.glob('%s/*_MSIL%s_*/GRANULE/*'%(infile, level)))
        
        # Where infile is a .SAFE file
        if '_MSIL%s_'%level in infile.split('/')[-1]: infiles_reduced.extend(glob.glob('%s/GRANULE/*'%infile))
        
        # Where infile is a specific granule 
        if len(infile.split('/')) >1 and infile.split('/')[-2] == 'GRANULE': infiles_reduced.extend(glob.glob('%s'%infile))
    
    # Strip repeats (in case)
    infiles_reduced = list(set(infiles_reduced))
    
    # Reduce input to infiles that match the tile (where specified)
    infiles_reduced = [infile for infile in infiles_reduced if ('_T%s'%tile in infile.split('/')[-1])]
    
    # Reduce input files to only L1C or L2A files
    infiles_reduced = [infile for infile in infiles_reduced if ('_MSIL%s_'%level in infile.split('/')[-3])]
    
    return infiles_reduced


def _sortScenes(scenes, by = 'tile'):
    '''
    Function to sort a list of scenes by tile, then by date. This is tidier, and reduces some artefacts in mosaics.
    
    Args:
        scenes: A list of sen2mosaic.LoadScene() Sentinel-2 objects
        by: Set to 'tile' to sort by tile then date, or 'date' to sort by date then tile
    Returns:
        A sorted list of scenes
    '''
    
    assert by in ['tile', 'date'], "Sentinel-2 scenes can only be sorted by 'tile' or by 'date'."
    
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


def loadSceneList(infiles, resolution = 20, md_dest = None, start = '20150101', end = datetime.datetime.today().strftime('%Y%m%d'), level = '2A', sort_by = None):
    """
    Function to load a list of infiles or all files in a directory as sen2moisac.LoadScene() objects.
    """
    
    # Prepare input string, or list of files
    source_files = prepInfiles(infiles, level)
    
    scenes = []
    for source_file in source_files:
        try:
            
            # Load scene
            scene = sen2mosaic.LoadScene(source_file, resolution = resolution)
            
            # Skip scene if conditions not met
            if md_dest is not None and scene.testInsideTile(md_dest) == False: continue
            if scene.testInsideDate(start = start, end = end) == False: continue
            
            scenes.append(scene)
        
        except Exception as e:
            
            print("WARNING: Error in loading scene %s with error '%s'. Continuing."%(source_file,str(e)))   
    
    # Optionally sort
    if sort_by is not None: scenes = _sortScenes(scenes, by = sort_by)
    
    return scenes
