from osgeo import gdal, ogr, osr
import numpy as np
import numpy.ma as ma
import os
import time
from utils.progress import progress_callback, progress_callback_quiet
from rasterUtils.gdalHelpers import getExtent, getIntersection, createSubsetDataframe, createClipGeoTransform, datatypeIsFloat, translateMaxValues


# Author: CFI
# TODO: Create a createDestinationFrame function

def clipRaster(inRaster, outRaster=None, referenceRaster=None, cutline=None,
               cutlineAllTouch=False, cropToCutline=True, srcNoDataValue=None,
               dstNoDataValue=None, quiet=False, align=True, bandToClip=None,
               calcBandStats=True, outputFormat='MEM'):
    ''' Clips a raster by either a reference raster, a cutline
        or both.

    Args:
        inRaster (URL or GDAL.DataFrame): The raster to clip.

    **kwargs:
        outRaster (URL): The name of the output raster. Only
        used when output format is not memory.

        referenceRaster (URL or GDAL.DataFrame): A reference
        raster from where to clip the extent of the inRaster.

        cutline (URL or OGR.DataFrame): A geometry used to cut
        the inRaster.

        cutlineAllTouch (Bool): Should all pixels that touch
        the cutline be included? False is only pixel centroids
        that fall within the geometry.

        cropToCutline (Bool): Should the output raster be
        clipped to the extent of the cutline geometry.

        srcNoDataValue (Number): Overwrite the nodata value of
        the source raster.

        dstNoDataValue (Number): Set a new nodata for the
        output array.

        quiet (Bool): Do not show the progressbars.

        align (Bool): Align the output pixels with the pixels
        in the input.

        bandToClip (Bool): Specify if only a specific band in
        the input raster should be clipped.

        outputFormat (String): Output of calculation. MEM is
        default, if outRaster is specified but MEM is selected,
        GTiff is used as outputformat.

    Returns:
        if MEM is selected as outputFormat a GDAL dataframe
        is returned, otherwise a URL reference to the created
        raster is returned.
    '''

    # Is the output format correct?
    if outRaster is None and outputFormat is not 'MEM':
        raise AttributeError("Either a reference raster or a cutline must be provided.")

    # If outRaster is specified, default to GTiff output format
    if outRaster is not None and outputFormat is 'MEM':
        outputFormat = 'GTiff'

    # Are none of either reference raster or cutline provided?
    if referenceRaster is None and cutline is None:
        raise AttributeError("Either a reference raster or a cutline must be provided.")

    # Read the supplied raster
    inputDataframe = gdal.Open(inRaster)

    # Throw error if GDAL cannot open the raster
    if inputDataframe is None:
        raise AttributeError(f"Unable to parse the input raster: {inRaster}")

    # Read the requested band. The NoDataValue might be: None.
    inputBand = inputDataframe.GetRasterBand(1)  # To read the datatype from

    # Read the attributes of the inputdataframe
    inputTransform = inputDataframe.GetGeoTransform()
    inputProjection = inputDataframe.GetProjection()
    inputProjectionRef = inputDataframe.GetProjectionRef()
    inputExtent = getExtent(inputDataframe)

    inputBandCount = inputDataframe.RasterCount  # Ensure compatability with multiband rasters
    outputBandCount = 1 if bandToClip is not None else inputBandCount

    ''' PREPARE THE NODATA VALUES '''
    # Get the nodata-values from either the raster or the function parameters
    inputNodataValue = inputBand.GetNoDataValue()

    if inputNodataValue is None and srcNoDataValue is not None:
        inputNodataValue = srcNoDataValue

    # If the destination nodata value is not set - set it to the input nodata value
    if dstNoDataValue is None:
        dstNoDataValue = inputNodataValue

    # If there is a cutline a nodata value must be set for the destination
    if cutline is not None and dstNoDataValue is None:
        dstNoDataValue = translateMaxValues(inputBand.DataType)
    if cutline is not None and inputNodataValue is None:
        inputNodataValue = translateMaxValues(inputBand.DataType)
        for i in range(inputBandCount):
            _inputBand = inputDataframe.GetRasterBand(i + 1)
            _inputBand.SetNoDataValue(inputNodataValue)

    ''' GDAL throws a warning whenever warpOptions are based to a function
        that has the 'MEM' format. However, it is necessary to do so because
        of the cutlineAllTouch feature.'''
    gdal.PushErrorHandler('CPLQuietErrorHandler')

    # If only one output band is requested it is necessary to create a subset
    # of the input data.
    if inputBandCount is not 1 and bandToClip is not None:
        # Set the origin to a subset band of the input raster
        origin = createSubsetDataframe(inputDataframe, bandToClip)
    else:
        # Subsets are not needed as the input only has one band.
        # Origin is then the input.
        origin = inputDataframe

    # Test the cutline
    if cutline is not None:
        # Test the cutline. This adds a tiny overhead, but is usefull to ensure
        # that the error messages are easy to understand.
        cutlineGeometry = ogr.Open(cutline)

        # Check if cutline was read properly.
        if cutlineGeometry is 0:         # GDAL returns 0 for warnings.
            print('Geometry read with warnings. Check your result.')
        elif cutlineGeometry is None:    # GDAL returns None for errors.
            raise RuntimeError("It was not possible to read the cutline geometry.") from None

        # Check whether it is a polygon or multipolygon
        layer = cutlineGeometry.GetLayer()
        vectorProjection = layer.GetSpatialRef()
        feat = layer.GetNextFeature()
        geom = feat.GetGeometryRef()
        geomType = geom.GetGeometryName()

        acceptedTypes = ['POLYGON', 'MULTIPOLYGON']
        if geomType not in acceptedTypes:
            raise RuntimeError("Only polygons or multipolygons are support as cutlines.") from None

        rasterProjection = osr.SpatialReference(inputProjection)

        # OGR has extents in different order than GDAL! minX minY maxX maxY
        vectorExtent = layer.GetExtent()
        vectorExtent = (vectorExtent[0], vectorExtent[2], vectorExtent[1], vectorExtent[3])

        if vectorProjection.ExportToProj4() is not rasterProjection.ExportToProj4():
            bottomLeft = ogr.Geometry(ogr.wkbPoint)
            topRight = ogr.Geometry(ogr.wkbPoint)

            bottomLeft.AddPoint(vectorExtent[0], vectorExtent[1])
            topRight.AddPoint(vectorExtent[2], vectorExtent[3])

            coordinateTransform = osr.CoordinateTransformation(vectorProjection, rasterProjection)
            bottomLeft.Transform(coordinateTransform)
            topRight.Transform(coordinateTransform)

            vectorExtent = (bottomLeft.GetX(), bottomLeft.GetY(), topRight.GetX(), topRight.GetY())

        # Test if the geometry and the raster intersect
        vectorIntersection = getIntersection(inputExtent, vectorExtent)

        if vectorIntersection is False:
            raise RuntimeError("The cutline did not intersect the input raster") from None

        # Free the memory again.
        cutlineGeometry = None
        layer = None
        feat = None
        geom = None
        geomType = None

    # Empty options to pass to GDAL.Warp. Since output is memory no compression
    # options are passed.
    options = ['INIT_DEST=NO_DATA']
    if cutlineAllTouch is True:
        options.append('CUTLINE_ALL_TOUCHED=TRUE')

    creationOptions = []
    if outputFormat is not 'MEM':
        if datatypeIsFloat(inputBand.DataType) is True:
            predictor = 3
        else:
            predictor = 2
        creationOptions = ['COMPRESS=DEFLATE', f'PREDICTOR={predictor}', 'NUM_THREADS=ALL_CPUS']

    # Create a GDAL driver to create dataframes in the right outputFormat
    driver = gdal.GetDriverByName(outputFormat)

    destinationName = outRaster if outRaster is not None else 'ignored'

    if referenceRaster is not None:
        # Read the reference raster
        referenceDataframe = gdal.Open(referenceRaster)

        # Throw error if GDAL cannot open the raster
        if referenceDataframe is None:
            raise AttributeError(f"Unable to parse the reference raster: {referenceRaster}")

        # Read the attributes of the referenceDataframe
        referenceTransform = referenceDataframe.GetGeoTransform()
        referenceProjection = referenceDataframe.GetProjection()

        # If the projections are the same there is no need to reproject.
        if osr.SpatialReference(inputProjection).ExportToProj4() == osr.SpatialReference(referenceProjection).ExportToProj4():
            referenceExtent = getExtent(referenceDataframe)
        else:
            progressbar = progress_callback_quiet
            if quiet is False:
                print(f"Reprojecting reference raster:")
                progressbar = progress_callback

            try:
                # Reproject the reference to match the input before cutting by extent
                reprojectedReferenceDataframe = gdal.Warp(
                    'ignored',
                    referenceDataframe,
                    format='MEM',
                    srcSRS=referenceProjection,
                    dstSRS=inputProjection,
                    multithread=True,
                    callback=progressbar,
                )
            except:
                raise RuntimeError("Error while Warping.") from None

            # Check if warped was successfull.
            if reprojectedReferenceDataframe is 0:         # GDAL returns 0 for warnings.
                print('Warping completed with warnings. Check your result.')
            elif reprojectedReferenceDataframe is None:    # GDAL returns None for errors.
                raise RuntimeError("Warping completed unsuccesfully.") from None

            referenceExtent = getExtent(reprojectedReferenceDataframe)

        # Calculate the bounding boxes and test intersection
        referenceIntersection = getIntersection(inputExtent, referenceExtent)

        # If they dont intersect, throw error
        if referenceIntersection is False:
            raise RuntimeError("The reference raster did not intersect the input raster") from None

        # Calculates the GeoTransform and rastersize from an extent and a geotransform
        referenceClipTransform = createClipGeoTransform(inputTransform, referenceIntersection)

        # Create the raster to serve as the destination for the warp
        destinationDataframe = driver.Create(
            destinationName,                          # Location of the saved raster, ignored if driver is memory.
            referenceClipTransform['RasterXSize'],    # Dataframe width in pixels (e.g. 1920px).
            referenceClipTransform['RasterYSize'],    # Dataframe height in pixels (e.g. 1280px).
            outputBandCount,                          # The number of bands required.
            inputBand.DataType,                       # Datatype of the destination
            creationOptions,                          # Compressions options for non-memory output.
        )

        destinationDataframe.SetGeoTransform(referenceClipTransform['Transform'])
        destinationDataframe.SetProjection(inputProjection)
        if inputNodataValue is not None:
            for i in range(inputBandCount):
                destinationBand = destinationDataframe.GetRasterBand(i + 1)
                destinationBand.SetNoDataValue(dstNoDataValue)
                destinationBand.FlushCache()

        progressbar = progress_callback_quiet
        if quiet is False:
            print(f"Clipping input raster:")
            progressbar = progress_callback

        ''' OBS: If cropToCutline and a reference raster are both provided. Crop to the
            reference raster instead of the cropToCutline.'''
        try:
            if cutline is None:
                warpSuccess = gdal.Warp(
                    destinationDataframe,
                    origin,
                    format=outputFormat,
                    targetAlignedPixels=align,
                    xRes=inputTransform[1],
                    yRes=inputTransform[5],
                    multithread=True,
                    srcNodata=inputNodataValue,
                    dstNodata=dstNoDataValue,
                    callback=progressbar,
                )
            else:
                warpSuccess = gdal.Warp(
                    destinationDataframe,
                    origin,
                    format=outputFormat,
                    targetAlignedPixels=align,
                    xRes=inputTransform[1],
                    yRes=inputTransform[5],
                    multithread=True,
                    srcNodata=inputNodataValue,
                    dstNodata=dstNoDataValue,
                    callback=progressbar,
                    cutlineDSName=cutline,
                    warpOptions=options,
                )
        except:
            raise RuntimeError("Error while Warping.") from None

        # Check if warped was successfull.
        if warpSuccess is 0:         # GDAL returns 0 for warnings.
            print('Warping completed with warnings. Check your result.')
        elif warpSuccess is None:    # GDAL returns None for errors.
            raise RuntimeError("Warping completed unsuccesfully.") from None

    # If a cutline is requested, a new DataFrame with the cut raster is created in memory
    elif cutline is not None:
        # Calculates the GeoTransform and rastersize from an extent and a geotransform
        vectorClipTransform = createClipGeoTransform(inputTransform, vectorIntersection)

        # Create the raster to serve as the destination for the warp
        destinationDataframe = driver.Create(
            destinationName,                        # Ignored as destination is memory.
            vectorClipTransform['RasterXSize'],     # Dataframe width in pixels (e.g. 1920px).
            vectorClipTransform['RasterYSize'],     # Dataframe height in pixels (e.g. 1280px).
            outputBandCount,                        # The number of bands required.
            inputBand.DataType,                     # Datatype of the destination
            creationOptions,                        # Compressions options for non-memory output.
        )
        destinationDataframe.SetGeoTransform(vectorClipTransform['Transform'])
        destinationDataframe.SetProjection(inputProjection)
        if inputNodataValue is not None:
            for i in range(inputBandCount):
                destinationBand = destinationDataframe.GetRasterBand(i + 1)
                destinationBand.SetNoDataValue(dstNoDataValue)
                destinationBand.FlushCache()

        progressbar = progress_callback_quiet
        if quiet is False:
            print(f"Warping the raster:")
            progressbar = progress_callback

        try:
            warpSuccess = gdal.Warp(
                destinationDataframe,
                origin,
                format=outputFormat,
                targetAlignedPixels=align,
                xRes=inputTransform[1],
                yRes=inputTransform[5],
                multithread=True,
                srcNodata=inputNodataValue,
                dstNodata=dstNoDataValue,
                callback=progressbar,
                cutlineDSName=cutline,
                cropToCutline=cropToCutline,
                warpOptions=options,
            )
        except:
            raise RuntimeError("Error while Warping.") from None

        # Check if warped was successfull.
        if warpSuccess is 0:         # GDAL returns 0 for warnings.
            print('Warping completed with warnings. Check your result.')
        elif warpSuccess is None:    # GDAL returns None for errors.
            raise RuntimeError("Warping completed unsuccesfully.") from None

    # Reenable the normal ErrorHandler.
    gdal.PopErrorHandler()

    # Close datasets again to free memory.
    inputDataframe = None
    referenceDataframe = None
    reprojectedReferenceDataframe = None

    if calcBandStats is True:
        for i in range(outputBandCount):
            destinationBand = destinationDataframe.GetRasterBand(i + 1).GetStatistics(0, 1)

    if outRaster is not None:
        destinationDataframe = None
        return os.path.abspath(outRaster)
    else:
        return destinationDataframe
