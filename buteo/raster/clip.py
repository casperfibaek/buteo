import os
import sys; sys.path.append('../../')
from osgeo import gdal, ogr, osr
from typing import Union
from buteo.raster.io import raster_to_reference, raster_to_metadata
from buteo.gdal_utils import (
    get_extent,
    create_geotransform,
    get_intersection,
    parse_projection,
    vector_to_reference,
)

def clip_raster(
    in_raster: Union[str, gdal.Dataset],
    clip_geom: Union[str, ogr.DataSource],
    out_path: Union[str, None]=None,
    crop_to_geom: bool=True,
    all_touch: bool=False,
    align_pixels: bool=True,
):
    # Verify inputs
    ref = raster_to_reference(in_raster)
    metadata = raster_to_metadata(ref)
    band_count = metadata["bands"]

    # Is the output format correct?
    if out_raster == None and output_format != "MEM":
        raise AttributeError("Either a reference raster or a cutline must be provided.")

    # If out_raster is specified, default to GTiff output format
    if out_raster != None and output_format == "MEM":
        output_format = "GTiff"

    # Are none of either reference raster or cutline provided?
    if reference_raster == None and cutline == None:
        raise AttributeError("Either a reference raster or a cutline must be provided.")

    # Read the supplied raster
    if isinstance(in_raster, gdal.Dataset):
        inputDataframe = in_raster
    else:
        try:
            inputDataframe = gdal.Open(in_raster)
        except:
            print("GDAL could not read: ", in_raster)
            exit()

    # Throw error if GDAL cannot open the raster
    if inputDataframe == None:
        raise AttributeError(f"Unable to parse the input raster: {in_raster}")

    # Read the requested band. The NoDataValue might be: None.
    inputBand = inputDataframe.GetRasterBand(1)  # To read the datatype from

    # Read the attributes of the inputdataframe
    inputTransform = inputDataframe.GetGeoTransform()
    inputProjection = inputDataframe.GetProjection()
    inputProjectionOSR = osr.SpatialReference(inputProjection)
    inputExtent = get_extent(inputDataframe)

    inputBandCount = (
        inputDataframe.RasterCount
    )  # Ensure compatability with multiband rasters
    outputBandCount = 1 if band_to_clip != None else inputBandCount

    # Create a GDAL driver to create dataframes in the right output_format
    driver = gdal.GetDriverByName(output_format)

    """ PREPARE THE NODATA VALUES """
    # Get the nodata-values from either the raster or the function parameters
    inputNodataValue = inputBand.GetNoDataValue()

    if inputNodataValue == None and src_nodata != None:
        inputNodataValue = src_nodata

    # If the destination nodata value is not set - set it to the input nodata value
    if dst_nodata == None:
        dst_nodata = inputNodataValue

    # If there is a cutline a nodata value must be set for the destination
    if cutline != None and dst_nodata == None:
        dst_nodata = translate_max_values(inputBand.DataType)
    if cutline != None and inputNodataValue == None:
        inputNodataValue = translate_max_values(inputBand.DataType)

    """ GDAL throws a warning whenever warpOptions are based to a function
        that has the 'MEM' format. However, it is necessary to do so because
        of the cutline_all_touch feature."""
    gdal.PushErrorHandler("CPLQuietErrorHandler")

    # If only one output band is requested it is necessary to create a subset
    # of the input data.
    if inputBandCount != 1 and band_to_clip != None:
        # Set the origin to a subset band of the input raster
        origin = create_subset_dataframe(inputDataframe, band_to_clip)
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
        if cutlineGeometry == 0:  # GDAL returns 0 for warnings.
            print("Geometry read with warnings. Check your result.")
        elif cutlineGeometry == None:  # GDAL returns None for errors.
            raise RuntimeError(
                "It was not possible to read the cutline geometry."
            ) from None

        # Check whether it is a polygon or multipolygon
        cutlineLayer = cutlineGeometry.GetLayer(0)

        # Check if layer contains features:
        vectorFeatureCount = cutlineLayer.GetFeatureCount()

        if vectorFeatureCount == 0:  # GDAL returns 0 for warnings.
            raise RuntimeError("Cutline geometry feature count is zero.") from None

        vectorProjection = cutlineLayer.GetSpatialRef()
        vectorProjectionOSR = osr.SpatialReference()
        vectorProjectionOSR.ImportFromWkt(str(vectorProjection))

        try:
            for feat in cutlineLayer:
                geom = feat.GetGeometryRef()
                geomType = geom.GetGeometryName()

                acceptedTypes = ["POLYGON", "MULTIPOLYGON"]
                if geomType not in acceptedTypes:
                    raise RuntimeError(
                        "Only polygons or multipolygons are support as cutlines."
                    ) from None
        except:
            Warning("Unable to get feature information. Attempting to continue..")

        # OGR has extents in different order than GDAL! minX minY maxX maxY
        vectorExtent = cutlineLayer.GetExtent()
        vectorExtent = (
            vectorExtent[0],
            vectorExtent[2],
            vectorExtent[1],
            vectorExtent[3],
        )

        if not vectorProjectionOSR.IsSame(inputProjectionOSR):
            bottomLeft = ogr.Geometry(ogr.wkbPoint)
            topRight = ogr.Geometry(ogr.wkbPoint)

            bottomLeft.AddPoint(vectorExtent[0], vectorExtent[1])
            topRight.AddPoint(vectorExtent[2], vectorExtent[3])
            coordinateTransform = osr.CoordinateTransformation(
                vectorProjection, inputProjectionOSR
            )
            bottomLeft.Transform(coordinateTransform)
            topRight.Transform(coordinateTransform)

            vectorExtent = (
                bottomLeft.GetX(),
                bottomLeft.GetY(),
                topRight.GetX(),
                topRight.GetY(),
            )

        # Test if the geometry and the raster intersect
        vectorIntersection = get_intersection(inputExtent, vectorExtent)

        if vectorIntersection == False:
            if quiet == False:
                print(
                    "The cutline did not intersect the input raster. Returning empty frame."
                )
            return False

        # Free the memory again.
        feat = None
        geom = None
        geomType = None

    # Empty options to pass to GDAL.Warp. Since output is memory no compression
    # options are passed.
    options = ["INIT_DEST=NO_DATA"]
    if cutline_all_touch == True:
        options.append("CUTLINE_ALL_TOUCHED=TRUE")

    creationOptions = []
    if output_format != "MEM":
        creationOptions = ["COMPRESS=LZW", "NUM_THREADS=ALL_CPUS", "BIGTIFF=TRUE"]

    destinationName = out_raster if out_raster != None else "ignored"

    if reference_raster != None:
        # Read the reference raster
        if isinstance(reference_raster, gdal.Dataset):
            referenceDataframe = reference_raster
        else:
            referenceDataframe = gdal.Open(reference_raster)

        # Throw error if GDAL cannot open the raster
        if referenceDataframe == None:
            raise AttributeError(
                f"Unable to parse the reference raster: {reference_raster}"
            )

        # Read the attributes of the referenceDataframe
        referenceTransform = referenceDataframe.GetGeoTransform()
        referenceProjection = referenceDataframe.GetProjection()
        referenceProjectionOSR = osr.SpatialReference()
        referenceProjectionOSR.ImportFromWkt(str(referenceProjection))

        # If the projections are the same there is no need to reproject.
        if referenceProjectionOSR.IsSame(inputProjectionOSR):
            referenceExtent = get_extent(referenceDataframe)
        else:
            progressbar = progress_callback_quiet
            if quiet == False:
                progressbar = create_progress_callback(1, "clipping")

            try:
                # Reproject the reference to match the input before cutting by extent
                reprojectedReferenceDataframe = gdal.Warp(
                    "ignored",
                    referenceDataframe,
                    format="MEM",
                    srcSRS=referenceProjection,
                    dstSRS=inputProjection,
                    multithread=True,
                    callback=progressbar,
                )
            except:
                raise RuntimeError("Error while Warping.") from None

            # Check if warped was successfull.
            if reprojectedReferenceDataframe == 0:  # GDAL returns 0 for warnings.
                print("Warping completed with warnings. Check your result.")
            elif reprojectedReferenceDataframe == None:  # GDAL returns None for errors.
                raise RuntimeError("Warping completed unsuccesfully.") from None

            referenceExtent = get_extent(reprojectedReferenceDataframe)

        # Calculate the bounding boxes and test intersection
        referenceIntersection = get_intersection(inputExtent, referenceExtent)
        referenceExtent = None

        # If they dont intersect, throw error
        if referenceIntersection == False:
            raise RuntimeError(
                "The reference raster did not intersect the input raster"
            ) from None

        # Calculates the GeoTransform and rastersize from an extent and a geotransform
        if scale_to_reference == True:
            referenceClipTransform = create_geotransform(
                referenceTransform, referenceIntersection
            )
        else:
            referenceClipTransform = create_geotransform(
                inputTransform, referenceIntersection
            )

        referenceIntersection = None

        # Create the raster to serve as the destination for the warp
        destinationDataframe = driver.Create(
            destinationName,  # Location of the saved raster, ignored if driver is memory.
            referenceClipTransform[
                "RasterXSize"
            ],  # Dataframe width in pixels (e.g. 1920px).
            referenceClipTransform[
                "RasterYSize"
            ],  # Dataframe height in pixels (e.g. 1280px).
            outputBandCount,  # The number of bands required.
            inputBand.DataType,  # Datatype of the destination
            creationOptions,  # Compressions options for non-memory output.
        )

        destinationDataframe.SetGeoTransform(referenceClipTransform["Transform"])
        destinationDataframe.SetProjection(inputProjection)
        if inputNodataValue != None:
            for i in range(inputBandCount):
                destinationBand = destinationDataframe.GetRasterBand(i + 1)
                destinationBand.SetNoDataValue(dst_nodata)
                destinationBand.FlushCache()
                destinationBand = None

        progressbar = progress_callback_quiet
        if quiet == False:
            progressbar = create_progress_callback(1, "clipping")

        """ OBS: If crop_to_cutline and a reference raster are both provided. Crop to the
            reference raster instead of the crop_to_cutline."""
        try:
            if cutline == None:
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
                    cutlineLayer=cutlineLayer,
                    cutlineWhere=cutlineWhere,
                    warpOptions=options,
                )
        except:
            raise RuntimeError("Error while Warping.") from None

        # Check if warped was successfull.
        if warpSuccess == 0:  # GDAL returns 0 for warnings.
            print("Warping completed with warnings. Check your result.")
        elif warpSuccess == None:  # GDAL returns None for errors.
            raise RuntimeError("Warping completed unsuccesfully.") from None

    # If a cutline is requested, a new DataFrame with the cut raster is created in memory
    elif cutline is not None:
        # Calculates the GeoTransform and rastersize from an extent and a geotransform
        vectorClipTransform = create_geotransform(inputTransform, vectorIntersection)

        if (
            vectorClipTransform["RasterXSize"] == 0
            or vectorClipTransform["RasterYSize"] == 0
        ):
            if quiet == False:
                print("The cutline only interesects at border. Returning empty frame.")
            return False

        # Create the raster to serve as the destination for the warp
        destinationDataframe = driver.Create(
            destinationName,  # Ignored as destination is memory.
            vectorClipTransform[
                "RasterXSize"
            ],  # Dataframe width in pixels (e.g. 1920px).
            vectorClipTransform[
                "RasterYSize"
            ],  # Dataframe height in pixels (e.g. 1280px).
            outputBandCount,  # The number of bands required.
            inputBand.DataType,  # Datatype of the destination
            creationOptions,  # Compressions options for non-memory output.
        )

        destinationDataframe.SetGeoTransform(vectorClipTransform["Transform"])
        destinationDataframe.SetProjection(inputProjection)
        if inputNodataValue != None:
            for i in range(inputBandCount):
                destinationBand = destinationDataframe.GetRasterBand(i + 1)
                destinationBand.SetNoDataValue(dst_nodata)
                destinationBand.FlushCache()
                destinationBand = None

        progressbar = progress_callback_quiet
        if quiet == False:
            progressbar = create_progress_callback(1, "clipping")

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
                cutlineLayer=cutlineLayer,
                cropToCutline=crop_to_cutline,
                cutlineWhere=cutlineWhere,
                warpOptions=options,
            )
        except:
            raise RuntimeError("Error while Warping.") from None

        # Check if warped was successfull.
        if warpSuccess == 0:  # GDAL returns 0 for warnings.
            print("Warping completed with warnings. Check your result.")
        elif warpSuccess == None:  # GDAL returns None for errors.
            raise RuntimeError("Warping completed unsuccesfully.") from None

    # Reenable the normal ErrorHandler.
    gdal.PopErrorHandler()

    # Close datasets again to free memory.
    inputDataframe = None
    referenceDataframe = None
    destinationName = None
    cutlineGeometry = None
    cutlineLayer = None
    in_raster = None
    reference_raster = None
    reprojectedReferenceDataframe = None
    inputBand = None
    origin = None
    driver = None

    if calc_band_stats == True:
        for i in range(outputBandCount):
            destinationBand = destinationDataframe.GetRasterBand(i + 1).GetStatistics(
                0, 1
            )

    if out_raster != None:
        destinationDataframe = None
        return os.path.abspath(out_raster)
    else:
        return destinationDataframe
