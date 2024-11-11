import sys; sys.path.append("../")
import os
import buteo as beo


FOLDER = "C:/Users/casper.fibaek/OneDrive - ESA/Desktop/s1_imagery_phileo/"
PATH = os.path.join(FOLDER, "DNK2_34_s1.tif")

bbox = beo.raster_to_metadata(PATH)["bbox_latlng"]
print(bbox)
