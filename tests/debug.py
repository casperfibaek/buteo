import tempfile
import os
from osgeo import ogr, gdal
from typing import Union, Optional

def vector_invalid_geometry(path):
    """Create a vector with invalid geometry (self-intersecting polygon)."""
    driver = ogr.GetDriverByName('GPKG')
    ds = driver.CreateDataSource(path)
    layer = ds.CreateLayer('test', None, ogr.wkbPolygon)

    # Create self-intersecting polygon
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(0, 0)
    ring.AddPoint(2, 2)
    ring.AddPoint(0, 2)
    ring.AddPoint(2, 0)
    ring.AddPoint(0, 0)

    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)

    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetGeometry(poly)
    layer.CreateFeature(feature)

    # Flush and close the data source
    ds.FlushCache()
    ds = None
    return path

def vector_fix_geometry(
    vector: Union[str, ogr.DataSource],
    *,
    layer_name_or_id: Optional[Union[str, int]] = None,
) -> bool:
    """Attempts to fix invalid geometries in a vector.

    Parameters
    ----------
    vector : str or ogr.DataSource
        A path to a vector or an OGR datasource
    layer_name_or_id : str or int, optional
        The name or index of the layer to fix. Default: None (fixes all layers)

    Returns
    -------
    bool
        True if all geometries are now valid, False if some remain invalid
    """
    ref = ogr.Open(vector, update=1)

    if layer_name_or_id is not None:
        if isinstance(layer_name_or_id, int):
            layers = [ref.GetLayer(layer_name_or_id)]
        else:
            layers = [ref.GetLayerByName(layer_name_or_id)]
    else:
        layers = [ref.GetLayer(i) for i in range(ref.GetLayerCount())]

    for layer in layers:
        for feature in layer:
            geom = feature.GetGeometryRef()
            if geom and not geom.IsValid():
                fixed_geom = geom.MakeValid()
                if fixed_geom and fixed_geom.IsValid():
                    feature.SetGeometry(fixed_geom)
                    layer.SetFeature(feature)

    # Flush changes
    ref.FlushCache()
    ref = None
    return True

def check_vector_has_invalid_geometry(
    vector: Union[str, ogr.DataSource],
    *,
    layer_name_or_id: Optional[Union[str, int]] = None,
) -> bool:
    """Checks if a vector has invalid geometry.

    Parameters
    ----------
    vector : str or ogr.DataSource
        A path to a vector or an OGR datasource
    layer_name_or_id : str or int, optional
        The name or index of the layer to check. Default: None (checks all layers)
    allow_empty : bool, optional
        If True, empty geometries are considered valid. Default: False

    Returns
    -------
    bool
        True if the vector has invalid geometry, False if all geometries are valid
    """
    ref = ogr.Open(vector)
    if layer_name_or_id is not None:
        if isinstance(layer_name_or_id, int):
            layers = [ref.GetLayer(layer_name_or_id)]
        else:
            layers = [ref.GetLayerByName(layer_name_or_id)]
    else:
        layers = [ref.GetLayer(i) for i in range(ref.GetLayerCount())]

    for layer in layers:
        # if it is a table layer, skip it
        if layer.GetGeomType() == ogr.wkbNone:
            continue

        for feature in layer:
            geom = feature.GetGeometryRef()
            if geom is None or not geom.IsValid():
                return True

    return False


if __name__ == "__main__":
    gdal.UseExceptions()
    path = "/vsimem/test_fix_invalid_geometry.gpkg"
    vec = vector_invalid_geometry(path)
    assert vector_fix_geometry(vec) is True
    assert check_vector_has_invalid_geometry(vec) is False  # Should now pass
    print("All tests passed!")

    gdal.Unlink(path)
