from osgeo import gdal


def merge_vrt(out_name, in_rasters, out_format='vrt', options=gdal.BuildVRTOptions(resampleAlg=gdal.GRA_NearestNeighbour, separate=True)):
    return gdal.BuildVRT(out_name, in_rasters, options=options)