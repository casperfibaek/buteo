# Standard library
import sys; sys.path.append("../")

# Standard library
import os

from buteo.eo.s2_utils import s2_l2a_get_metadata


FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s2_data/"

s2_file = os.path.join(FOLDER, "S2A_MSIL2A_20220107T090351_N0301_R007_T36UVB_20220107T120342.zip")

bob = s2_l2a_get_metadata(s2_file)

import pdb; pdb.set_trace()