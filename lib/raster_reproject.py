import os
import numpy as np
from pyproj import CRS
from osgeo import gdal, osr
from lib.utils_core import datatype_is_float
from lib.raster_io import array_to_raster



def reproject(in_raster, out_raster=None, reference_raster=None, target_projection=None, resampling=0, output_format='MEM', quiet=True, compress=True):
    '''
    '''
    
    # Is the output format correct?
    if out_raster is None and output_format != 'MEM':
        raise AttributeError("If output_format is not MEM, out_raster must be defined")

    # If out_raster is specified, default to GTiff output format
    if out_raster is not None and output_format == 'MEM':
        output_format = 'GTiff'

    if out_raster is None:
        out_raster = 'ignored'   # This is necessary as GDAL expects a string no matter what.
    else:
        assert os.path.isdir(os.path.dirname(out_raster)), f'Output folder does not exists: {out_raster}'
    
    assert reference_raster is None or target_projection is None, 'reference_raster and target_epsg cannot be applied at the same time.'

    if isinstance(in_raster, gdal.Dataset):  # Dataset already GDAL dataframe.
        source_raster = in_raster
    else:
        try:
            source_raster = gdal.Open(in_raster, gdal.GA_ReadOnly)
        except:
            try: 
                if isinstance(in_raster, np.ndarray):
                    source_raster = array_to_raster(in_raster, reference_raster=reference_raster)
                else:
                    raise Exception('Unable to transform in_raster.')
            except:
                raise Exception('Unable to read in_raster.')

    # Gather reference information
    if reference_raster is not None:
        if isinstance(reference_raster, gdal.Dataset):  # Dataset already GDAL dataframe.
            target_projection = CRS.from_wkt(reference_raster.GetProjection())
        else:
            try:
                target_projection = CRS.from_wkt(gdal.Open(reference_raster, gdal.GA_ReadOnly).GetProjection())
            except:
                raise Exception('Unable to read reference_raster.')
    else:
        try:
            target_projection = CRS.from_epsg(target_projection)
        except:
            try:
                target_projection = CRS.from_wkt(target_projection)
            except:
                try:
                    if isinstance(target_projection, CRS):
                        target_projection = target_projection
                    else:
                        raise Exception('Unable to transform target_projection')
                except:
                    raise Exception('Unable to read target_projection')
    
    driver = gdal.GetDriverByName(output_format)
    datatype = source_raster.GetRasterBand(1).DataType

    # If the output is not memory, set compression options.
    creation_options = []
    if compress is True:
        if output_format != 'MEM':
            if datatype_is_float(datatype) is True:
                predictor = 3  # Float predictor
            else:
                predictor = 2  # Integer predictor
            creation_options = ['COMPRESS=DEFLATE', f'PREDICTOR={predictor}', 'NUM_THREADS=ALL_CPUS', 'BIGTIFF=YES']
    

    og_projection_osr = osr.SpatialReference() 
    og_projection_osr.ImportFromWkt(source_raster.GetProjection())
    dst_projection_osr = osr.SpatialReference()
    dst_projection_osr.ImportFromWkt(target_projection.to_wkt())

    og_transform = source_raster.GetGeoTransform()

    og_x_size = source_raster.RasterXSize
    og_y_size = source_raster.RasterYSize

    coord_transform = osr.CoordinateTransformation(og_projection_osr, dst_projection_osr)
    
    o_ulx, xres, xskew, o_uly, yskew, yres = og_transform
    o_lrx = o_ulx + (og_x_size * xres)
    o_lry = o_uly + (og_y_size * yres)

    og_col = (o_lrx - o_ulx)
    og_row = (o_uly - o_lry)

    ulx, uly, ulz = coord_transform.TransformPoint(float(o_ulx), float(o_uly))
    urx, ury, urz = coord_transform.TransformPoint(float(o_lrx), float(o_uly))
    lrx, lry, lrz = coord_transform.TransformPoint(float(o_lrx), float(o_lry))
    llx, lly, llz = coord_transform.TransformPoint(float(o_ulx), float(o_lry))
    
    dst_col = max(lrx, urx) - min(llx, ulx)
    dst_row = max(ury, uly) - min(lry, lly)

    cols = int((dst_col / og_col) * og_x_size)
    rows = int((dst_row / og_row) * og_y_size)
    
    dst_pixel_width = dst_col / cols
    dst_pixel_height = dst_row / rows

    dst_transform = (min(ulx, llx), dst_pixel_width, -0.0, max(uly, ury), 0.0, -dst_pixel_height)

    destination_dataframe = driver.Create(
        out_raster,
        cols,
        rows,
        1,
        datatype,
        creation_options
    )

    destination_dataframe.SetProjection(target_projection.to_wkt())
    destination_dataframe.SetGeoTransform(dst_transform)

    gdal.Warp(
        destination_dataframe,
        source_raster,
        format=output_format,
        multithread=True,
        srcSRS=og_projection_osr.ExportToWkt(),
        dstSRS=target_projection.to_wkt(),
    )
    # gdal.ReprojectImage(source_raster, destination_dataframe, og_projection_osr.ExportToWkt(), target_projection.to_wkt(), resampling)

    destination_dataframe.FlushCache()

    if output_format == 'MEM':
        return destination_dataframe
    else:
        destination_dataframe = None
        return out_raster
