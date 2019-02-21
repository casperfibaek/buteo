from osgeo import gdal, osr
import numpy as np
import numpy.ma as ma
import os
from utils import datatype_is_float, numpy_to_gdal_datatypes, progress_callback, progress_callback_quiet


def arrayToRaster(array, out_raster=None, reference_raster=None, output_format='MEM',
                  top_left=None, pixel_size=None, projection=None, calc_band_stats=True,
                  src_nodata=None, resample=False, quiet=False):
    ''' Turns a numpy array into a gdal dataframe or exported
        as a raster. If no reference is specified, the following
        must be provided: topLeft coordinates, pixelSize such as:
        (10, 10), projection in proj4 format and output raster size.

        The datatype of the raster will match the datatype of the input
        numpy array.

        OBS: If WGS84 lat long is specified as the projection the pixel
        sizes must be in degrees.

    Args:
        in_raster (URL or GDAL.DataFrame): The raster to clip.

    **kwargs:
        reference_raster (URL or GDAL.DataFrame): A reference
        raster from where to clip the extent of the inRaster.

        out_raster (URL): The name of the output raster. Only
        used when output format is not memory.

        output_format (String): Output of calculation. MEM is
        default, if out_raster is specified but MEM is selected,
        GTiff is used as output_format.

        top_left (List): The coordinates for the topleft corner
        of the raster. Such as: [600000, 520000].

        pixel_size (List): The width and height of the pixels in
        the destination raster. Such as: [10, 10].

        projection (String): A proj4 string matching the projection
        of the destination raster. Such as: "+proj=utm +zone=32
            +ellps=WGS84 +datum=WGS84 +units=m +no_defs".

        calc_band_stats (Bool): Calculate band statistics and add
        them to the image. Might increase processing time.

        src_nodata (Number): Overwrite the nodata value of
        the source raster.

        resample (Bool): Resample the input raster to match the
        reference if they do not overlap.

        quiet (Bool): Do not show the progressbars for warping.

    Returns:
        If the output format is memory outpus a GDAL dataframe
        containing the data contained in the array. Otherwise a
        raster is created and the return value is the URL string
        pointing to the created raster.
    '''

    ''' **********************************************************
        STEP (1): Verify the input data.
        ********************************************************** '''

    # Is the output format correct?
    if out_raster is None and output_format != 'MEM':
        raise AttributeError("Either a reference raster or a cutline must be provided.")

    # If out_raster is specified, default to GTiff output format
    if out_raster is not None and output_format == 'MEM':
        output_format = 'GTiff'

    if out_raster is None:
        out_raster = 'ignored'   # This is necessary as GDAL expects a string no matter what.

    if reference_raster is None and (
        top_left is None or
        pixel_size is None or
        projection is None
    ):
        raise AttributeError("If no reference_raster is provided. top_left, pixel_size, projection and raster_size are all required.")

    if reference_raster is not None and (
        top_left is not None or
        pixel_size is not None or
        projection is not None
    ):
        print('WARNING: Only the values from the reference_raster will be used.')

    ''' **********************************************************
        STEP (2): Setup local values and ready data.
    ********************************************************** '''

    # The data that will be written to the raster
    # If the data is not a numpy array, make it.
    data = array if isinstance(array, np.ndarray) else np.array(array)
    datatype = numpy_to_gdal_datatypes(data.dtype)

    if data.ndim != 2:
        raise AttributeError("The input raster must be 2-dimensional")

    reference = {} if reference_raster is not None else None

    # Gather reference information
    if reference_raster is not None:

        if isinstance(reference_raster, gdal.Dataset):  # Dataset alreadt GDAL dataframe.
            reference['dataframe'] = reference_raster
        else:
            reference['dataframe'] = gdal.Open(reference_raster)

        # Throw error if GDAL cannot open the raster
        if reference['dataframe'] is None:
            raise AttributeError(f"Unable to parse the reference raster: {reference_raster}")

        reference['transform'] = reference['dataframe'].GetGeoTransform()
        reference['projection'] = reference['dataframe'].GetProjection()
        reference['x_size'] = reference['dataframe'].RasterXSize
        reference['y_size'] = reference['dataframe'].RasterYSize
        reference['pixel_width'] = reference['transform'][1]
        reference['pixel_height'] = reference['transform'][5]
        reference['x_top_left'] = reference['transform'][0]
        reference['y_top_left'] = reference['transform'][3]
        reference['bands'] = [reference['dataframe'].GetRasterBand(1)]
        reference['nodata'] = reference['bands'][0].GetNoDataValue()
    else:
        reference['x_size'] = data.shape[0]
        reference['y_size'] = data.shape[1]
        reference['pixel_width'] = pixel_size[0]
        reference['pixel_height'] = pixel_size[1]
        reference['x_top_left'] = top_left[0]
        reference['y_top_left'] = top_left[1]
        reference['nodata'] = None
        reference['transform'] = [
            reference['x_top_left'], reference['pixel_width'], 0,
            reference['y_top_left'], 0, -reference['pixel_height'],
        ]
        reference['projection'] = osr.SpatialReference()
        reference['projection'].ImportFromProj4(projection)
        reference['projection'].ExportToWkt()
        reference['projection'] = str(reference['projection'])

    input_nodata = reference['nodata']

    # Ready the nodata values
    if ma.is_masked(data) is True:
        if input_nodata is not None:
            ma.masked_equal(data, input_nodata)
            data.set_fill_value(input_nodata)
        input_nodata = data.get_fill_value()
        data = data.filled()

    # Weird "double" issue with GDAL and numpy. Cast to float or int
    if input_nodata is not None:
        input_nodata = float(input_nodata)
        if (input_nodata).is_integer() is True:
            input_nodata = int(input_nodata)

    # If the output is not memory, set compression options.
    options = []
    if output_format != 'MEM':
        if datatype_is_float(datatype) is True:
            predictor = 3  # Float predictor
        else:
            predictor = 2  # Integer predictor
        options = ['COMPRESS=DEFLATE', f'PREDICTOR={predictor}', 'NUM_THREADS=ALL_CPUS']

    ''' **********************************************************
        STEP (3): The business logic.
    ********************************************************** '''
    destination = {}

    if resample is False:
        destination['driver'] = gdal.GetDriverByName(output_format)
        destination['name'] = out_raster
        destination['options'] = options
    else:
        destination['driver'] = gdal.GetDriverByName('MEM')
        destination['name'] = 'ignored'
        destination['options'] = []

    destination['dataframe'] = destination['driver'].Create(
        destination['name'],
        data.shape[1],
        data.shape[0],
        1,
        datatype,
        destination['options']
    )

    # Test if the scale is correct and set transform
    if data.shape[0] == reference['x_size'] and data.shape[1] == reference['y_size']:
        destination['dataframe'].SetGeoTransform(reference['transform'])
    else:
        destination['pixel_width'] = (reference['x_size'] / data.shape[1]) * reference['pixel_width']
        destination['pixel_height'] = (reference['y_size'] / data.shape[0]) * reference['pixel_height']
        destination['dataframe'].SetGeoTransform([
            reference['x_top_left'],
            destination['pixel_width'],
            0,
            reference['y_top_left'],
            0,
            destination['pixel_height'],
        ])

    destination['dataframe'].SetProjection(reference['projection'])
    destination['bands'] = [destination['dataframe'].GetRasterBand(1)]
    destination['bands'][0].WriteArray(data)

    if input_nodata is not None:
        destination['bands'][0].SetNoDataValue(input_nodata)

    # If it is necessary to resample to fit the reference:
    if resample is True and (data.shape[0] != reference['x_size'] or data.shape[1] != reference['y_size']):
        resampled_destination = {'driver': gdal.GetDriverByName(output_format)}
        resampled_destination['dataframe'] = resampled_destination['driver'].Create(
            out_raster, reference['x_size'], reference['y_size'], 1, datatype, options)

        resampled_destination['dataframe'].SetGeoTransform(reference['transform'])
        resampled_destination['dataframe'].SetGeoTransform(reference['projection'])

        progressbar = progress_callback_quiet
        if quiet is False:
            print(f"Warping input array:")
            progressbar = progress_callback

        gdal.PushErrorHandler('CPLQuietErrorHandler')

        try:
            warp_sucess = gdal.Warp(
                resampled_destination['dataframe'],
                destination['dataframe'],
                format=output_format,
                xRes=reference['x_size'],
                yRes=reference['y_size'],
                srcSRS=reference['projection'],
                multithread=True,
                callback=progressbar,
            )
        except:
            raise RuntimeError("Error while Warping.") from None

        gdal.PopErrorHandler()

        # Check if warped was successfull.
        if warp_sucess == 0:         # GDAL returns 0 for warnings.
            print('Warping completed with warnings. Check your result.')
        elif warp_sucess is None:    # GDAL returns None for errors.
            raise RuntimeError("Warping completed unsuccesfully.") from None

        # Clear memory
        destination = None

        if calc_band_stats is True:
            resampled_destination['dataframe'].GetRasterBand(1).GetStatistics(0, 1)

        reference_raster = None

        # Return the resampled destination raster
        if output_format != 'MEM':
            resampled_destination.FlushCache()
            resampled_destination = None
            return os.path.abspath(out_raster)
        else:
            return resampled_destination

    if calc_band_stats is True:
        destination['bands'][0].GetStatistics(0, 1)

    reference_raster = None

    # Return the destination raster
    if output_format != 'MEM':
        destination.FlushCache()
        destination = None
        return os.path.abspath(out_raster)
    else:
        return destination
