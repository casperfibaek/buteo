""" This module contains utility functions for testing purposes. """

# Standard library
import random
from uuid import uuid4

# External
import numpy as np
from osgeo import gdal, osr, ogr


def create_sample_raster(
        width=10,
        height=10,
        bands=1,
        pixel_width=1,
        pixel_height=1,
        x_min=None,
        y_max=None,
        epsg_code=4326,
        datatype=gdal.GDT_Byte,
        nodata=None,
    ):
    """
    Create a sample raster file for testing purposes. (GTiff)
    
    Valid epsg codes:
        3857: Web Mercator
        4326: WGS84
        25832: ETRS89-UTM32N
    """
    raster_path = f"/vsimem/mem_raster_{uuid4().int}.tif"
    driver = gdal.GetDriverByName("GTiff")
    raster = driver.Create(raster_path, width, height, bands, datatype)

    if y_max is None:
        y_max = height * pixel_height
    if x_min is None:
        x_min = 0

    raster.SetGeoTransform((x_min, pixel_width, 0, y_max, 0, -pixel_height))

    for band in range(1, bands + 1):
        raster.GetRasterBand(band).WriteArray(np.random.randint(0, 255, (height, width), dtype=np.uint8))

    if nodata is not None:
        for band in range(1, bands + 1):
            raster.GetRasterBand(band).SetNoDataValue(float(nodata))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_code)
    raster.SetProjection(srs.ExportToWkt())
    raster.FlushCache()
    raster = None

    return raster_path


def create_sample_vector(
    geom_type="polygon",
    num_features=10,
    epsg_code=4326,
    attribute_data=None,
    n_layers=1,
):
    """
    Create a sample vector dataset for testing purposes. (GPKG)

    Parameters
    ----------
    geom_type : int, optional
        The geometry type for the vector dataset. Default is ogr.wkbPolygon.

    num_features : int, optional
        The number of features to create in the vector dataset. Default is 10.

    epsg_code : int, optional
        The EPSG code for the spatial reference system. Default is 4326 (WGS84).
        Valid epsg codes:
            3857: Web Mercator
            4326: WGS84
            25832: ETRS89-UTM32N

    attribute_data : list of dicts, optional
        A list of dictionaries containing attribute data for the features.
        Default is None.

    Returns
    -------
    str
        The path to the in-memory vector dataset.
    """
    if geom_type == "polygon":
        geom_type = ogr.wkbPolygon
    elif geom_type == "point":
        geom_type = ogr.wkbPoint
    else:
        raise ValueError("Invalid geometry type. Must be either 'polygon' or 'point'.")

    vector_path = f"/vsimem/mem_vector_{uuid4().int}.gpkg"
    driver = ogr.GetDriverByName("GPKG")
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_code)

    vector_ds = driver.CreateDataSource(vector_path)
    for i in range(n_layers):
        layer = vector_ds.CreateLayer(f"sample_layer_{str(i + 1)}", srs, geom_type)

        if attribute_data:
            for attr in attribute_data:
                field_defn = ogr.FieldDefn(attr["name"], attr["type"])
                layer.CreateField(field_defn)

        for i in range(num_features):
            feature = ogr.Feature(layer.GetLayerDefn())

            if geom_type == ogr.wkbPolygon:
                ring = ogr.Geometry(ogr.wkbLinearRing)
                x, y = random.uniform(-180, 180), random.uniform(-90, 90)
                ring.AddPoint(x, y)
                ring.AddPoint(x + 1, y)
                ring.AddPoint(x + 1, y + 1)
                ring.AddPoint(x, y + 1)
                ring.AddPoint(x, y)
                polygon = ogr.Geometry(ogr.wkbPolygon)
                polygon.AddGeometry(ring)
                feature.SetGeometry(polygon)

            elif geom_type == ogr.wkbPoint:
                x, y = random.uniform(-180, 180), random.uniform(-90, 90)
                point = ogr.Geometry(ogr.wkbPoint)
                point.AddPoint(x, y)
                feature.SetGeometry(point)

            if attribute_data:
                for attr in attribute_data:
                    feature.SetField(attr["name"], attr["values"][i])

            layer.CreateFeature(feature)

        layer.SyncToDisk()

    vector_ds.FlushCache()
    vector_ds = None
    layer = None

    return vector_path
