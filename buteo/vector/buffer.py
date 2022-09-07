"""
### Clip vectors to other geometries ###

Clip vector files with other geometries. Can come from rasters or vectors.
"""

# Standard library
import sys; sys.path.append("../../")

# External
from osgeo import ogr

# Internal
from buteo.utils import gdal_utils
from buteo.vector import core_vector


"""
TODO:
    Warning if 4326
    All layers
    Memory files etc.
"""
def buffer_vector(vector, distance, out_path, layer_idx=0):
    """ Buffer vector """
    read = core_vector.open_vector(vector)

    vector_metadata = core_vector._vector_to_metadata(read)
    vector_layername = vector_metadata["layers"][layer_idx]["layer_name"]
    vector_layer = read.GetLayer(vector_layername)

    driver = ogr.GetDriverByName(gdal_utils.path_to_driver_vector(out_path))
    destination = driver.CreateDataSource(out_path)
    destination.CopyLayer(vector_layer, vector_layername, ["OVERWRITE=YES"])

    sql = f"update {vector_layername} set geom=ST_Buffer(geom, {distance})"

    destination.ExecuteSQL(sql, dialect="SQLITE")

    if destination is None:
        raise Exception("Error while running intersect.")

    destination.FlushCache()

    return out_path

folder = "/home/casper/Desktop/buteo/geometry_and_rasters/"
vector_path = folder + "beirut_city_utm36.gpkg"
outpath = folder + "beirut_city_utm36_buffered.gpkg"

buffer_vector(vector_path, 100.0, outpath)
