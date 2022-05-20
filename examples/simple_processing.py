import sys; sys.path.append("../")

from buteo.raster.reproject import reproject_raster

folder = "C:/Users/caspe/Desktop/test_rasters/"
raster = folder + "B12_20m.tif"

reproject_raster(raster, 4326, folder + "B12_20m_4326.tif")