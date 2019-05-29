import os
import numpy as np
import numpy.ma as ma
from osgeo import gdal, ogr, osr
import utils


def clip_raster(in_raster, out_raster=None, reference_raster=None, cutline=None,
                cutline_all_touch=False, crop_to_cutline=True, cutlineWhere=None, src_nodata=None,
                dst_nodata=None, quiet=False, align=True, band_to_clip=None,
                calc_band_stats=True, output_format='MEM'):
    ''' Clips a raster by either a reference raster, a cutline
        or both.

    Args:
        in_raster (URL or GDAL.DataFrame): The raster to clip.

    **kwargs:
        out_raster (URL): The name of the output raster. Only
        used when output format is not memory.

        reference_raster (URL or GDAL.DataFrame): A reference
        raster from where to clip the extent of the in_raster.

        cutline (URL or OGR.DataFrame): A geometry used to cut
        the in_raster.

        cutline_all_touch (Bool): Should all pixels that touch
        the cutline be included? False is only pixel centroids
        that fall within the geometry.

        crop_to_cutline (Bool): Should the output raster be
        clipped to the extent of the cutline geometry.

        src_nodata (Number): Overwrite the nodata value of
        the source raster.

        dst_nodata (Number): Set a new nodata for the
        output array.

        quiet (Bool): Do not show the progressbars.

        align (Bool): Align the output pixels with the pixels
        in the input.

        band_to_clip (Bool): Specify if only a specific band in
        the input raster should be clipped.

        output_format (String): Output of calculation. MEM is
        default, if out_raster is specified but MEM is selected,
        GTiff is used as output_format.

    Returns:
        if MEM is selected as output_format a GDAL dataframe
        is returned, otherwise a URL reference to the created
        raster is returned.
    '''

    # Is the output format correct?
    if out_raster is None and output_format != 'MEM':
        raise AttributeError("Either a reference raster or a cutline must be provided.")

    # If out_raster is specified, default to GTiff output format
    if out_raster is not None and output_format == 'MEM':
        output_format = 'GTiff'

    # Are none of either reference raster or cutline provided?
    if reference_raster is None and cutline is None:
        raise AttributeError("Either a reference raster or a cutline must be provided.")

    # Read the supplied raster
    if isinstance(in_raster, gdal.Dataset):
        inputDataframe = in_raster
    else:
        inputDataframe = gdal.Open(in_raster)

    # Throw error if GDAL cannot open the raster
    if inputDataframe is None:
        raise AttributeError(f"Unable to parse the input raster: {in_raster}")

    # Read the requested band. The NoDataValue might be: None.
    inputBand = inputDataframe.GetRasterBand(1)  # To read the datatype from

    # Read the attributes of the inputdataframe
    inputTransform = inputDataframe.GetGeoTransform()
    inputProjection = inputDataframe.GetProjection()
    inputProjectionRef = inputDataframe.GetProjectionRef()
    inputExtent = utils.get_extent(inputDataframe)

    inputBandCount = inputDataframe.RasterCount  # Ensure compatability with multiband rasters
    outputBandCount = 1 if band_to_clip is not None else inputBandCount

    ''' PREPARE THE NODATA VALUES '''
    # Get the nodata-values from either the raster or the function parameters
    inputNodataValue = inputBand.GetNoDataValue()

    if inputNodataValue is None and src_nodata is not None:
        inputNodataValue = src_nodata

    # If the destination nodata value is not set - set it to the input nodata value
    if dst_nodata is None:
        dst_nodata = inputNodataValue

    # If there is a cutline a nodata value must be set for the destination
    if cutline is not None and dst_nodata is None:
        dst_nodata = utils.translate_max_values(inputBand.DataType)
    if cutline is not None and inputNodataValue is None:
        inputNodataValue = utils.translate_max_values(inputBand.DataType)
        # TODO: FIX
        # for i in range(inputBandCount):
        #     _inputBand = inputDataframe.GetRasterBand(i + 1)
        #     _inputBand.SetNoDataValue(inputNodataValue)

    ''' GDAL throws a warning whenever warpOptions are based to a function
        that has the 'MEM' format. However, it is necessary to do so because
        of the cutline_all_touch feature.'''
    gdal.PushErrorHandler('CPLQuietErrorHandler')

    # If only one output band is requested it is necessary to create a subset
    # of the input data.
    if inputBandCount != 1 and band_to_clip is not None:
        # Set the origin to a subset band of the input raster
        origin = utils.create_subset_dataframe(inputDataframe, band_to_clip)
    else:
        # Subsets are not needed as the input only has one band.
        # Origin is then the input.
        origin = inputDataframe

    # Test the cutline
    if cutline is not None:
        # Test the cutline. This adds a tiny overhead, but is usefull to ensure
        # that the error messages are easy to understand.
        if isinstance(cutline, ogr.DataSource):
            cutlineGeometry = cutline
        else:
            cutlineGeometry = ogr.Open(cutline)

        # Check if cutline was read properly.
        if cutlineGeometry == 0:         # GDAL returns 0 for warnings.
            print('Geometry read with warnings. Check your result.')
        elif cutlineGeometry is None:    # GDAL returns None for errors.
            raise RuntimeError("It was not possible to read the cutline geometry.") from None

        # Check whether it is a polygon or multipolygon
        layer = cutlineGeometry.GetLayer()
        vectorProjection = layer.GetSpatialRef()
        vectorProjectionOSR = osr.SpatialReference()
        vectorProjectionOSR.ImportFromWkt(str(vectorProjection))

        feat = layer.GetNextFeature()
        geom = feat.GetGeometryRef()
        geomType = geom.GetGeometryName()

        acceptedTypes = ['POLYGON', 'MULTIPOLYGON']
        if geomType not in acceptedTypes:
            raise RuntimeError("Only polygons or multipolygons are support as cutlines.") from None

        rasterProjectionOSR = osr.SpatialReference(inputProjection)

        # OGR has extents in different order than GDAL! minX minY maxX maxY
        vectorExtent = layer.GetExtent()
        vectorExtent = (vectorExtent[0], vectorExtent[2], vectorExtent[1], vectorExtent[3])

        if not vectorProjectionOSR.IsSame(rasterProjectionOSR):
            bottomLeft = ogr.Geometry(ogr.wkbPoint)
            topRight = ogr.Geometry(ogr.wkbPoint)

            bottomLeft.AddPoint(vectorExtent[0], vectorExtent[1])
            topRight.AddPoint(vectorExtent[2], vectorExtent[3])

            coordinateTransform = osr.CoordinateTransformation(vectorProjection, inputProjection)
            bottomLeft.Transform(coordinateTransform)
            topRight.Transform(coordinateTransform)

            vectorExtent = (bottomLeft.GetX(), bottomLeft.GetY(), topRight.GetX(), topRight.GetY())

        # Test if the geometry and the raster intersect
        vectorIntersection = utils.get_intersection(inputExtent, vectorExtent)

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
    if cutline_all_touch is True:
        options.append('CUTLINE_ALL_TOUCHED=TRUE')

    creationOptions = []
    if output_format != 'MEM':
        if utils.datatype_is_float(inputBand.DataType) is True:
            predictor = 3
        else:
            predictor = 2
        creationOptions = ['COMPRESS=DEFLATE', f'PREDICTOR={predictor}', 'NUM_THREADS=ALL_CPUS']

    # Create a GDAL driver to create dataframes in the right output_format
    driver = gdal.GetDriverByName(output_format)

    destinationName = out_raster if out_raster is not None else 'ignored'

    if reference_raster is not None:
        # Read the reference raster

        if isinstance(reference_raster, gdal.Dataset):
            referenceDataframe = reference_raster
        else:
            referenceDataframe = gdal.Open(reference_raster)

        # Throw error if GDAL cannot open the raster
        if referenceDataframe is None:
            raise AttributeError(f"Unable to parse the reference raster: {reference_raster}")

        # Read the attributes of the referenceDataframe
        referenceTransform = referenceDataframe.GetGeoTransform()
        referenceProjection = referenceDataframe.GetProjection()
        referenceProjectionOSR = osr.SpatialReference()
        referenceProjectionOSR.ImportFromWkt(str(referenceProjection))

        # If the projections are the same there is no need to reproject.
        if inputProjection.IsSame(referenceProjectionOSR):
            referenceExtent = utils.get_extent(referenceDataframe)
        else:
            progressbar = utils.progress_callback_quiet
            if quiet is False:
                progressbar = utils.create_progress_callback(1, 'clipping')

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

            referenceExtent = utils.get_extent(reprojectedReferenceDataframe)

        # Calculate the bounding boxes and test intersection
        referenceIntersection = utils.get_intersection(inputExtent, referenceExtent)
        referenceExtent = None

        # If they dont intersect, throw error
        if referenceIntersection is False:
            raise RuntimeError("The reference raster did not intersect the input raster") from None

        # Calculates the GeoTransform and rastersize from an extent and a geotransform
        referenceClipTransform = utils.create_geotransform(inputTransform, referenceIntersection)

        referenceIntersection = None

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
                destinationBand.SetNoDataValue(dst_nodata)
                destinationBand.FlushCache()
                destinationBand = None

        progressbar = utils.progress_callback_quiet
        if quiet is False:
            progressbar = utils.create_progress_callback(1, 'clipping')

        ''' OBS: If crop_to_cutline and a reference raster are both provided. Crop to the
            reference raster instead of the crop_to_cutline.'''
        try:
            if cutline is None:
                warpSuccess = gdal.Warp(
                    destinationDataframe,
                    origin,
                    format=output_format,
                    targetAlignedPixels=align,
                    xRes=inputTransform[1],
                    yRes=inputTransform[5],
                    multithread=True,
                    srcNodata=inputNodataValue,
                    dstNodata=dst_nodata,
                    callback=progressbar,
                )
            else:
                warpSuccess = gdal.Warp(
                    destinationDataframe,
                    origin,
                    format=output_format,
                    targetAlignedPixels=align,
                    xRes=inputTransform[1],
                    yRes=inputTransform[5],
                    multithread=True,
                    srcNodata=inputNodataValue,
                    dstNodata=dst_nodata,
                    callback=progressbar,
                    cutlineDSName=cutline,
                    cutlineWhere=cutlineWhere,
                    warpOptions=options,
                )
        except:
            raise RuntimeError("Error while Warping.") from None

        # Check if warped was successfull.
        if warpSuccess == 0:         # GDAL returns 0 for warnings.
            print('Warping completed with warnings. Check your result.')
        elif warpSuccess is None:    # GDAL returns None for errors.
            raise RuntimeError("Warping completed unsuccesfully.") from None

    # If a cutline is requested, a new DataFrame with the cut raster is created in memory
    elif cutline is not None:
        # Calculates the GeoTransform and rastersize from an extent and a geotransform
        vectorClipTransform = utils.create_geotransform(inputTransform, vectorIntersection)

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
                destinationBand.SetNoDataValue(dst_nodata)
                destinationBand.FlushCache()
                destinationBand = None

        progressbar = utils.progress_callback_quiet
        if quiet is False:
            progressbar = utils.create_progress_callback(1, 'clipping')

        try:
            warpSuccess = gdal.Warp(
                destinationDataframe,
                origin,
                format=output_format,
                targetAlignedPixels=align,
                xRes=inputTransform[1],
                yRes=inputTransform[5],
                multithread=True,
                srcNodata=inputNodataValue,
                dstNodata=dst_nodata,
                callback=progressbar,
                cutlineDSName=cutline,
                cropToCutline=crop_to_cutline,
                cutlineWhere=cutlineWhere,
                warpOptions=options,
            )
        except:
            raise RuntimeError("Error while Warping.") from None

        # Check if warped was successfull.
        if warpSuccess == 0:         # GDAL returns 0 for warnings.
            print('Warping completed with warnings. Check your result.')
        elif warpSuccess is None:    # GDAL returns None for errors.
            raise RuntimeError("Warping completed unsuccesfully.") from None

    # Reenable the normal ErrorHandler.
    gdal.PopErrorHandler()

    # Close datasets again to free memory.

    inputDataframe = None
    referenceDataframe = None
    destinationName = None
    cutlineGeometry = None
    layer = None
    in_raster = None
    reference_raster = None
    reprojectedReferenceDataframe = None
    inputBand = None
    origin = None
    driver = None
    _inputBand = None

    if calc_band_stats is True:
        for i in range(outputBandCount):
            destinationBand = destinationDataframe.GetRasterBand(i + 1).GetStatistics(0, 1)

    if out_raster is not None:
        destinationDataframe = None
        return os.path.abspath(out_raster)
    else:
        return destinationDataframe
