import sys
sys.path.append('../lib')

from raster_io import clip_raster


clip_raster('C:\\Users\\caspe\\Desktop\\Ghana Data\\VH_Clipped.tif', 'C:\\Users\\caspe\\Desktop\\Ghana Data\\VH_Clipped_Land.tif', cutline='C:\\Users\\caspe\\Desktop\\Ghana Data\\Ghana_landarea2.shp')
