"""
Typings for project types.

TODO:
    - Improve the naming convention
"""

from osgeo import osr, ogr
from typing import Dict, Tuple, Union, List, Any, Optional, TypedDict


Number = Union[float, int]

Expanded_extents = TypedDict(
    "Expanded_extents",
    {
        "extent_wkt": str,
        "extent_datasource": ogr.DataSource,
        "extent_geom": ogr.Geometry,
        "extent_latlng": List[Number],
        "extent_gdal_warp_latlng": List[Number],
        "extent_ogr_latlng": List[Number],
        "extent_dict_latlng": Dict[str, Number],
        "extent_wkt_latlng": str,
        "extent_datasource_latlng": ogr.DataSource,
        "extent_geom_latlng": ogr.Geometry,
        "extent_geojson": str,
        "extent_geojson_dict": Dict[str, Any],
    },
)


Metadata_raster = TypedDict(
    "Metadata_raster",
    {
        "path": str,
        "basename": str,
        "name": str,
        "ext": str,
        "transform": List[Number],
        "in_memory": bool,
        "projection": str,
        "projection_osr": osr.SpatialReference,
        "width": int,
        "height": int,
        "band_count": int,
        "driver": str,
        "size": List[int],
        "shape": Tuple[Number, Number, Number],
        "pixel_width": Number,
        "pixel_height": Number,
        "x_min": Number,
        "y_max": Number,
        "x_max": Number,
        "y_min": Number,
        "datatype": str,
        "datatype_gdal": str,
        "datatype_gdal_raw": int,
        "nodata_value": Optional[Number],
        "has_nodata": bool,
        "is_vector": bool,
        "is_raster": bool,
        "extent": List[Number],
        "extent_ogr": List[Number],
        "extent_gdal_warp": List[Number],
        "extent_dict": Dict[str, Number],
        "extent_wkt": Optional[str],
        "extent_datasource": Optional[ogr.DataSource],
        "extent_geom": Optional[ogr.Geometry],
        "extent_latlng": Optional[List[Number]],
        "extent_gdal_warp_latlng": Optional[List[Number]],
        "extent_ogr_latlng": Optional[List[Number]],
        "extent_dict_latlng": Optional[Dict[str, Number]],
        "extent_wkt_latlng": Optional[str],
        "extent_datasource_latlng": Optional[ogr.DataSource],
        "extent_geom_latlng": Optional[ogr.Geometry],
        "extent_geojson": Optional[str],
        "extent_geojson_dict": Optional[Dict[str, Any]],
    },
)


# Used for comparing two rasters.
Metadata_raster_comp = TypedDict(
    "Metadata_raster_comp",
    {
        "projection": Optional[str],
        "pixel_width": Optional[Number],
        "pixel_height": Optional[Number],
        "x_min": Optional[Number],
        "y_max": Optional[Number],
        "transform": Optional[List[Number]],
        "width": Optional[int],
        "height": Optional[int],
        "datatype": Optional[str],
        "nodata_value": Optional[Number],
    },
)


Metadata_vector_layer = TypedDict(
    "Metadata_vector_layer",
    {
        "layer_name": str,
        "x_min": Number,
        "x_max": Number,
        "y_min": Number,
        "y_max": Number,
        "column_fid": str,
        "column_geom": str,
        "feature_count": int,
        "projection": str,
        "projection_osr": osr.SpatialReference,
        "geom_type": str,
        "geom_type_ogr": int,
        "field_count": int,
        "field_names": List[str],
        "field_types": List[str],
        "field_types_ogr": List[int],
        "extent": List[Number],
        "extent_ogr": List[Number],
        "extent_dict": Dict[str, Number],
        "extent_wkt": Optional[str],
        "extent_datasource": Optional[ogr.DataSource],
        "extent_geom": Optional[ogr.Geometry],
        "extent_latlng": Optional[List[Number]],
        "extent_gdal_warp_latlng": Optional[List[Number]],
        "extent_ogr_latlng": Optional[List[Number]],
        "extent_dict_latlng": Optional[Dict[str, Number]],
        "extent_wkt_latlng": Optional[str],
        "extent_datasource_latlng": Optional[ogr.DataSource],
        "extent_geom_latlng": Optional[ogr.Geometry],
        "extent_geojson": Optional[str],
        "extent_geojson_dict": Optional[Dict[str, Any]],
    },
)


Metadata_vector = TypedDict(
    "Metadata_vector",
    {
        "path": str,
        "basename": str,
        "name": str,
        "ext": str,
        "in_memory": bool,
        "projection": str,
        "projection_osr": osr.SpatialReference,
        "driver": str,
        "x_min": Number,
        "y_max": Number,
        "x_max": Number,
        "y_min": Number,
        "is_vector": bool,
        "is_raster": bool,
        "layer_count": int,
        "layers": List[Metadata_vector_layer],
        "extent": List[Number],
        "extent_ogr": List[Number],
        "extent_gdal_warp": List[Number],
        "extent_dict": Dict[str, Number],
        "extent_wkt": Optional[str],
        "extent_datasource": Optional[ogr.DataSource],
        "extent_geom": Optional[ogr.Geometry],
        "extent_latlng": Optional[List[Number]],
        "extent_gdal_warp_latlng": Optional[List[Number]],
        "extent_ogr_latlng": Optional[List[Number]],
        "extent_dict_latlng": Optional[Dict[str, Number]],
        "extent_wkt_latlng": Optional[str],
        "extent_datasource_latlng": Optional[ogr.DataSource],
        "extent_geom_latlng": Optional[ogr.Geometry],
        "extent_geojson": Optional[str],
        "extent_geojson_dict": Optional[Dict[str, Any]],
    },
)