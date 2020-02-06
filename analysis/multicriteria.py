import sys
sys.path.append('../lib')

from raster_io import raster_to_array, array_to_raster

andrekomplan2 = raster_to_array('C:\\Users\\caspe\\Desktop\\scaled\\andrekomplan2.tif', fill_value=0, src_nodata=0, filled=True) * 4 * 0.012
prio_natura212 = raster_to_array('C:\\Users\\caspe\\Desktop\\scaled\\prio_natura212.tif', fill_value=0, src_nodata=0, filled=True) * 10 * 0.173
ovrig_natura212 = raster_to_array('C:\\Users\\caspe\\Desktop\\scaled\\ovrig_natura212.tif', fill_value=0, src_nodata=0, filled=True) * 10 * 0.133
unesco_area2 = raster_to_array('C:\\Users\\caspe\\Desktop\\scaled\\unesco_area2.tif', fill_value=0, src_nodata=0, filled=True) * 8 * 0.163
unesco_buf2 = raster_to_array('C:\\Users\\caspe\\Desktop\\scaled\\unesco_buf2.tif', fill_value=0, src_nodata=0, filled=True) * 1 *0.006
fred_omr2 = raster_to_array('C:\\Users\\caspe\\Desktop\\scaled\\fred_omr2.tif', fill_value=0, src_nodata=0, filled=True) * 8 * 0.067
fred_forslag2 = raster_to_array('C:\\Users\\caspe\\Desktop\\scaled\\fred_forslag2.tif', fill_value=0, src_nodata=0, filled=True) * 6 * 0.009
besnatur2 = raster_to_array('C:\\Users\\caspe\\Desktop\\scaled\\besnatur2.tif', fill_value=0, src_nodata=0, filled=True) * 6 * 0.034
besnat_sammen2 = raster_to_array('C:\\Users\\caspe\\Desktop\\scaled\\besnat_sammen2.tif', fill_value=0, src_nodata=0, filled=True) * 8 * 0.043
fundarealbesk2 = raster_to_array('C:\\Users\\caspe\\Desktop\\scaled\\fundarealbesk2.tif', fill_value=0, src_nodata=0, filled=True) * 5 * 0.018
fundfortid2 = raster_to_array('C:\\Users\\caspe\\Desktop\\scaled\\fundfortid2.tif', fill_value=0, src_nodata=0, filled=True) * 9 * 0.073
fundbesk_sam2 = raster_to_array('C:\\Users\\caspe\\Desktop\\scaled\\fundbesk_sam2.tif', fill_value=0, src_nodata=0, filled=True) * 7 * 0.025
bygn_fred2 = raster_to_array('C:\\Users\\caspe\\Desktop\\scaled\\bygn_fred2.tif', fill_value=0, src_nodata=0, filled=True) * 6 * 0.057
boligomr20012 = raster_to_array('C:\\Users\\caspe\\Desktop\\scaled\\boligomr20012.tif', fill_value=0, src_nodata=0, filled=True) * 8 * 0.154
boligo500100012 = raster_to_array('C:\\Users\\caspe\\Desktop\\scaled\\boligo500100012.tif', fill_value=0, src_nodata=0, filled=True) * 4 * 0.008
boligom20050012 = raster_to_array('C:\\Users\\caspe\\Desktop\\scaled\\boligom20050012.tif', fill_value=0, src_nodata=0, filled=True) * 6 * 0.014
foreneligfred2 = raster_to_array('C:\\Users\\caspe\\Desktop\\scaled\\foreneligfred2.tif', fill_value=0, src_nodata=0, filled=True) * 2 * 0.011

multicriteria = andrekomplan2 + prio_natura212 + ovrig_natura212 + unesco_area2 + unesco_buf2 + fred_omr2 + fred_forslag2 + besnatur2 + besnat_sammen2 + fundarealbesk2 + fundfortid2 + fundbesk_sam2 + bygn_fred2 + boligomr20012 + boligo500100012 + boligom20050012 + foreneligfred2

array_to_raster(multicriteria, out_raster='C:\\Users\\caspe\\Desktop\\scaled\\multicriteria3.tif', reference_raster='C:\\Users\\caspe\\Desktop\\scaled\\fred_omr2.tif', src_nodata=0, dst_nodata=None)
