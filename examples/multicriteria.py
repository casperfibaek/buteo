import sys
sys.path.append('../lib/base')

from raster_io import raster_to_array, array_to_raster

base = 'C:\\Users\\caspe\\Desktop\\Projects\\multicriteriaAnalysis_vejdirektorat'

andrekomplan2 = raster_to_array(f'{base}\\andrekomplan2.tif', fill_value=0, src_nodata=0, filled=True) * 4 * 0.013
unesco_area2 = raster_to_array(f'{base}\\unesco_area2.tif', fill_value=0, src_nodata=0, filled=True) * 8 * 0.169
unesco_buf2 = raster_to_array(f'{base}\\unesco_buf2.tif', fill_value=0, src_nodata=0, filled=True) * 1 *0.086
besnatur2 = raster_to_array(f'{base}\\besnatur2.tif', fill_value=0, src_nodata=0, filled=True) * 6 * 0.038
besnat_sammen2 = raster_to_array(f'{base}\\besnat_sammen2.tif', fill_value=0, src_nodata=0, filled=True) * 8 * 0.049
fundarealbesk2 = raster_to_array(f'{base}\\fundarealbesk2.tif', fill_value=0, src_nodata=0, filled=True) * 5 * 0.019
fundfortid2 = raster_to_array(f'{base}\\fundfortid2.tif', fill_value=0, src_nodata=0, filled=True) * 9 * 0.089
fundbesk_sam2 = raster_to_array(f'{base}\\fundbesk_sam2.tif', fill_value=0, src_nodata=0, filled=True) * 7 * 0.026
bygn_fred2 = raster_to_array(f'{base}\\bygn_fred2.tif', fill_value=0, src_nodata=0, filled=True) * 6 * 0.067
boligo500100012 = raster_to_array(f'{base}\\boligo500100012.tif', fill_value=0, src_nodata=0, filled=True) * 4 * 0.010
foreneligfred2 = raster_to_array(f'{base}\\foreneligfred2.tif', fill_value=0, src_nodata=0, filled=True) * 2 * 0.011
boligomr_komplan000500mbuf = raster_to_array(f'{base}\\boligomr_komplan000500mbuff_v2.tif', fill_value=0, src_nodata=0, filled=True) * 8 * 0.182
fred_fredforslag = raster_to_array(f'{base}\\fred_fredforslag_v2.tif', fill_value=0, src_nodata=0, filled=True) * 9 * 0.084
natura2000_korr = raster_to_array(f'{base}\\natura2000_korr_v2.tif', fill_value=0, src_nodata=0, filled=True) * 10 * 0.237

multicriteria = andrekomplan2 + unesco_area2 + unesco_buf2 + besnatur2 + besnat_sammen2 + fundarealbesk2 + fundfortid2 + fundbesk_sam2 + bygn_fred2 + boligo500100012 + foreneligfred2 + boligomr_komplan000500mbuf + fred_fredforslag + natura2000_korr

array_to_raster(multicriteria, out_raster=f'{base}\\multicriteria_victor.tif', reference_raster=f'{base}\\andrekomplan2.tif', src_nodata=0, dst_nodata=None)
