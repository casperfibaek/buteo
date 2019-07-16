from .lib import shift_raster

rast = 'E:\\sentinel_1_data\\ghana\\slc\\step2\\wet_season.data\\coh_VV_15Jun2019_21Jun2019.img'
out_rast = 'E:\\sentinel_1_data\\ghana\\slc\\step2\\wet_season.data\\coh_VV_15Jun2019_21Jun2019_moved.tif'

shift_raster(rast, out_rast, 80, 0)
