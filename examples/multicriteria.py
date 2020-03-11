import sys; sys.path.append('..')
import numpy as np
from lib.raster_io import raster_to_array, array_to_raster

base = '/mnt/c/Users/caspe/Desktop/Projects/multicriteriaAnalysis_vejdirektorat/'

layers = np.array([
  raster_to_array(f'{base}andrekomplan2.tif', fill_value=0, src_nodata=0, filled=True) * 4 * 0.013,
  raster_to_array(f'{base}unesco_area2.tif', fill_value=0, src_nodata=0, filled=True) * 8 * 0.169,
  raster_to_array(f'{base}unesco_buf2.tif', fill_value=0, src_nodata=0, filled=True) * 1 *0.086,
  raster_to_array(f'{base}besnatur2.tif', fill_value=0, src_nodata=0, filled=True) * 6 * 0.038,
  raster_to_array(f'{base}besnat_sammen2.tif', fill_value=0, src_nodata=0, filled=True) * 8 * 0.049,
  raster_to_array(f'{base}fundarealbesk2.tif', fill_value=0, src_nodata=0, filled=True) * 5 * 0.019,
  raster_to_array(f'{base}fundfortid2.tif', fill_value=0, src_nodata=0, filled=True) * 9 * 0.089,
  raster_to_array(f'{base}fundbesk_sam2.tif', fill_value=0, src_nodata=0, filled=True) * 7 * 0.026,
  raster_to_array(f'{base}bygn_fred2.tif', fill_value=0, src_nodata=0, filled=True) * 6 * 0.067,
  raster_to_array(f'{base}boligo500100012.tif', fill_value=0, src_nodata=0, filled=True) * 4 * 0.010,
  raster_to_array(f'{base}foreneligfred2.tif', fill_value=0, src_nodata=0, filled=True) * 2 * 0.011,
  raster_to_array(f'{base}boligomr_komplan000500mbuff_v2.tif', fill_value=0, src_nodata=0, filled=True) * 8 * 0.182,
  raster_to_array(f'{base}fred_fredforslag_v2.tif', fill_value=0, src_nodata=0, filled=True) * 9 * 0.084,
  raster_to_array(f'{base}natura2000_korr_v2.tif', fill_value=0, src_nodata=0, filled=True) * 10 * 0.237,
])

array_to_raster(
  np.sum(layers, axis=0),
  out_raster=f'{base}multicriteria_victor2.tif',
  reference_raster=f'{base}andrekomplan2.tif',
  src_nodata=0,
  dst_nodata=None,
)
