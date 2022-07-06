import sys; sys.path.append("../")
import buteo
from glob import glob

folder = "/home/casper/Desktop/data/sentinel2_images/S2B_MSIL2A_20220629T081609_N0400_R121_T36SYC_20220629T102008.SAFE/GRANULE/L2A_T36SYC_A027746_20220629T082256/IMG_DATA/R60m/"

img = glob(folder + "*.jp2")[0]

import pdb; pdb.set_trace()
