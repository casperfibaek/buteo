import sys; sys.path.append('../utils')
import os
import numpy.ma as ma
from osgeo import gdal
from core import datatype_is_float, copy_dataframe, progress_callback_quiet, create_progress_callback, translate_resample_method


def resample(in_raster, out_raster=None, reference_raster=None, reference_band_number=1,
             target_size=None, output_format='MEM', method='nearest', quiet=False):
    ''' Resample an input raster to match a reference raster or
        a target size. The target size is in the same unit as
        the input raster. If the unit is wgs84 - beware that
        the target should be in degrees.

    Args:
        in_raster (URL or GDAL.DataFrame): The raster to clip.

    **kwargs:
        out_raster (URL): The name of the output raster. Only
        used when output format is not memory.

        reference_raster (URL or GDAL.DataFrame): A reference
        raster from where to clip the extent of the in_raster.

        reference_band_number (Integer): The number of the band in
        the reference to use as a target.

        target_size (List): The target pixel size of the destination.
        e.g. [10, 10] for 10m resolution if the map unit is meters.

        output_format (String): Output of calculation. MEM is
        default, if out_raster is specified but MEM is selected,
        GTiff is used as output_format.

        method (String): The method to use for resampling. By default
        nearest pixel is used. Supports bilinear, cubic and so on.
        For more details look at the GDAL reference.

        quiet (Bool): Do not show the progressbars for warping.

    Returns:
        If the output format is memory outpus a GDAL dataframe
        containing the resampled raster. Otherwise a
        raster is created and the return value is the URL string
        pointing to the created raster.
    '''

    if target_size is None and reference_raster is None:
        raise ValueError('Either target_size or a reference must be provided.')

    if out_raster is not None and output_format == 'MEM':
        output_format = 'GTiff'

    if out_raster is None and output_format != 'MEM':
        raise AttributeError("Either a reference raster or a cutline must be provided.")

    if out_raster is None:
        out_raster = 'ignored'

    if isinstance(in_raster, gdal.Dataset):
        inputDataframe = in_raster
    else:
        inputDataframe = gdal.Open(in_raster)

    # Throw error if GDAL cannot open the raster
    if inputDataframe is None:
        raise AttributeError(f"Unable to parse the input raster: {in_raster}")

    driver = gdal.GetDriverByName(output_format)

    inputTransform = inputDataframe.GetGeoTransform()
    inputPixelWidth = inputTransform[1]
    inputPixelHeight = inputTransform[5]
    inputProjection = inputDataframe.GetProjection()
    inputBandCount = inputDataframe.RasterCount
    inputBand = inputDataframe.GetRasterBand(1)
    inputDatatype = inputBand.DataType
    inputNodataValue = inputBand.GetNoDataValue()

    if output_format == 'MEM':
        options = []
    else:
        if datatype_is_float(inputDatatype) is True:
            predictor = 3
        else:
            predictor = 2
        options = ['COMPRESS=DEFLATE', f'PREDICTOR={predictor}', 'NUM_THREADS=ALL_CPUS']

    # Test if the same size is requested.
    if target_size is not None:
        if abs(inputPixelWidth) == abs(target_size[0]) and abs(inputPixelHeight) == abs(target_size[1]):
            copy = copy_dataframe(inputDataframe, out_raster, output_format)
            copy.FlushCache()

            if output_format == 'MEM':
                return copy
            else:
                return os.path.abspath(out_raster)

    if reference_raster is not None:
        if isinstance(reference_raster, gdal.Dataset):
            referenceDataframe = reference_raster
        else:
            referenceDataframe = gdal.Open(reference_raster)

        # Throw error if GDAL cannot open the raster
        if referenceDataframe is None:
            raise AttributeError(f"Unable to parse the reference raster: {reference_raster}")

        referenceTransform = referenceDataframe.GetGeoTransform()
        referenceProjection = referenceDataframe.GetProjection()
        referenceXSize = referenceDataframe.RasterXSize
        referenceYSize = referenceDataframe.RasterYSize
        referencePixelWidth = referenceTransform[1]
        referencePixelHeight = referenceTransform[5]
        referenceBand = referenceDataframe.GetRasterBand(reference_band_number)
        referenceDatatype = referenceBand.DataType

        # Test if the reference size and the input size are the same
        if abs(inputPixelWidth) == abs(referencePixelWidth) and abs(inputPixelHeight) == abs(referencePixelHeight):
            copy = copy_dataframe(inputDataframe, out_raster, output_format)
            copy.FlushCache()

            if output_format == 'MEM':
                return copy
            else:
                return os.path.abspath(out_raster)
        else:
            destination = driver.Create(out_raster, referenceXSize, referenceYSize, inputBandCount, referenceDatatype, options)
            destination.SetProjection(referenceProjection)
            destination.SetGeoTransform(referenceTransform)

        gdal.PushErrorHandler('CPLQuietErrorHandler')

        progressbar = progress_callback_quiet
        if quiet is False:
            progressbar = create_progress_callback(1, 'resampling')

        try:
            warpSuccess = gdal.Warp(
                destination, in_raster,
                format=output_format,
                multithread=True,
                targetAlignedPixels=True,
                xRes=referenceTransform[1],
                yRes=referenceTransform[5],
                srcSRS=inputProjection,
                dstSRS=referenceProjection,
                srcNodata=inputNodataValue,
                dstNodata=inputNodataValue,
                warpOptions=options,
                resampleAlg=translate_resample_method(method),
                callback=progressbar,
            )
        except:
            raise RuntimeError("Error while Warping.") from None

        gdal.PopErrorHandler()

        # Check if warped was successfull.
        if warpSuccess == 0:         # GDAL returns 0 for warnings.
            print('Warping completed with warnings. Check your result.')
        elif warpSuccess is None:    # GDAL returns None for errors.
            raise RuntimeError("Warping completed unsuccesfully.") from None

        if output_format == 'MEM':
            return destination
        else:
            return os.path.abspath(out_raster)
    else:

        inputXSize = inputDataframe.RasterXSize
        inputYSize = inputDataframe.RasterYSize
        xRatio = inputPixelWidth / target_size[0]
        yRatio = inputPixelHeight / target_size[1]
        xPixels = abs(round(xRatio * inputXSize))
        yPixels = abs(round(yRatio * inputYSize))

        destination = driver.Create(out_raster, xPixels, yPixels, inputBandCount, inputDatatype, options)
        destination.SetProjection(inputProjection)
        destination.SetGeoTransform([
            inputTransform[0], target_size[0], inputTransform[2],
            inputTransform[3], inputTransform[4], -target_size[1],
        ])
        if inputNodataValue is not None:
            destination.SetNoDataValue(inputNodataValue)

        gdal.PushErrorHandler('CPLQuietErrorHandler')

        progressbar = progress_callback_quiet
        if quiet is False:
            progressbar = create_progress_callback(1, 'resampling')

        try:
            warpSuccess = gdal.Warp(
                destination, in_raster,
                format=output_format,
                multithread=True,
                targetAlignedPixels=True,
                xRes=target_size[0],
                yRes=target_size[1],
                srcSRS=inputProjection,
                dstSRS=inputProjection,
                srcNodata=inputNodataValue,
                dstNodata=inputNodataValue,
                warpOptions=options,
                resampleAlg=translate_resample_method(method),
                callback=progressbar,
            )

        except:
            raise RuntimeError("Error while Warping.") from None

        gdal.PopErrorHandler()

        # Check if warped was successfull.
        if warpSuccess == 0:         # GDAL returns 0 for warnings.
            print('Warping completed with warnings. Check your result.')
        elif warpSuccess is None:    # GDAL returns None for errors.
            raise RuntimeError("Warping completed unsuccesfully.") from None

        if output_format == 'MEM':
            return destination
        else:
            return os.path.abspath(out_raster)
