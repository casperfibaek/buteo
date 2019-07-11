import sys
sys.path.append('../lib')

from zscores import calc_zscores

calc_zscores('D:\\T33UVB_20190529T101039_B04_20m.jp2', out_raster='D:\\z_mad.tif', mad=True)
