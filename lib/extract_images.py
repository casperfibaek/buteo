import sys; sys.path.append('..');
import numpy as np
from lib.raster_io import raster_to_array, array_to_raster, raster_to_metadata

# def extract_tiles(image, grid):

folder = "C:\\Users\\caspe\\Desktop\\Paper_2_StruturalDensity\\"
in_raster = "testing_image_320m.tif"



if __name__ == "__main__":
    ras = raster_to_array(folder + in_raster)
    import pdb; pdb.set_trace()