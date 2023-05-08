# Standard library
import sys; sys.path.append("../../")
from typing import Union, Optional, List

# External
import numpy as np
from osgeo import ogr

# Internal
from buteo.utils import (
    utils_io,
    utils_base,
    utils_gdal,
    utils_bbox,
    utils_path,
)
from buteo.vector.metadata import _vector_to_metadata
from buteo.vector.core_vector import _vector_open



def _vector_add_shapes_in_place(
    vector: Union[ogr.DataSource, str],
    shapes: Optional[List[str]] = None,
    prefix: str = "",
    verbose: bool = False,
) -> str:
    """ Internal. """
    assert isinstance(vector, (ogr.DataSource, str)), "vector must be a vector layer or path to one."
    assert isinstance(shapes, (list, tuple)) or shapes is None, "shapes must be a list of shapes."
    assert isinstance(prefix, str), "prefix must be a string."

    all_shapes = ["area", "perimeter", "ipq", "hull", "compactness", "centroid"]

    if shapes is None:
        shapes = all_shapes
    else:
        for shape in shapes:
            if shape not in all_shapes:
                raise ValueError(f"{shape} is not a valid shape.")

    datasource = _vector_open(vector)
    out_path = utils_gdal._get_path_from_dataset(datasource, dataset_type="vector")
    metadata = _vector_to_metadata(datasource)

    for index in range(metadata["layer_count"]):
        vector_current_fields = metadata["layers"][index]["field_names"]
        vector_layer = datasource.GetLayer(index)

        vector_layer.StartTransaction()

        # Add missing fields
        for attribute in shapes:
            if attribute == "centroid":
                if "centroid_x" not in vector_current_fields:
                    field_defn = ogr.FieldDefn(f"{prefix}centroid_x", ogr.OFTReal)
                    vector_layer.CreateField(field_defn)

                if "centroid_y" not in vector_current_fields:
                    field_defn = ogr.FieldDefn(f"{prefix}centroid_y", ogr.OFTReal)
                    vector_layer.CreateField(field_defn)

            elif attribute not in vector_current_fields:
                field_defn = ogr.FieldDefn(f"{prefix}{attribute}", ogr.OFTReal)
                vector_layer.CreateField(field_defn)

        vector_feature_count = vector_layer.GetFeatureCount()

        if verbose:
            utils_base.progress(0, vector_feature_count, name="shape")

        for i in range(vector_feature_count):
            vector_feature = vector_layer.GetNextFeature()

            try:
                vector_geom = vector_feature.GetGeometryRef()
            except RuntimeWarning:
                # vector_geom.Buffer(0)
                raise RuntimeWarning("Invalid geometry at : ", i) from None

            if vector_geom is None:
                raise RuntimeError("Invalid geometry. Could not fix.")

            centroid = vector_geom.Centroid()
            vector_area = vector_geom.GetArea()
            vector_perimeter = vector_geom.Boundary().Length()

            if "ipq" or "compact" in shapes:
                vector_ipq = 0
                if vector_perimeter != 0:
                    vector_ipq = (4 * np.pi * vector_area) / vector_perimeter ** 2

            if "centroid" in shapes:
                vector_feature.SetField(f"{prefix}centroid_x", centroid.GetX())
                vector_feature.SetField(f"{prefix}centroid_y", centroid.GetY())

            if "hull" in shapes or "compact" in shapes:
                vector_hull = vector_geom.ConvexHull()
                hull_area = vector_hull.GetArea()
                hull_peri = vector_hull.Boundary().Length()
                hull_ratio = float(vector_area) / float(hull_area)
                compactness = np.sqrt(float(hull_ratio) * float(vector_ipq))

            if "area" in shapes:
                vector_feature.SetField(f"{prefix}area", vector_area)
            if "perimeter" in shapes:
                vector_feature.SetField(f"{prefix}perimeter", vector_perimeter)
            if "ipq" in shapes:
                vector_feature.SetField(f"{prefix}ipq", vector_ipq)
            if "hull" in shapes:
                vector_feature.SetField(f"{prefix}hull_area", hull_area)
                vector_feature.SetField(f"{prefix}hull_peri", hull_peri)
                vector_feature.SetField(f"{prefix}hull_ratio", hull_ratio)
            if "compact" in shapes:
                vector_feature.SetField(f"{prefix}compact", compactness)

            vector_layer.SetFeature(vector_feature)

            if verbose:
                utils_base.progress(i, vector_feature_count, name="shape")

        vector_layer.CommitTransaction()

    return out_path


def vector_add_shapes_in_place(
    vector: Union[str, ogr.DataSource, List[Union[str, ogr.DataSource]]],
    shapes: Optional[List[str]] = None,
    prefix: str = "",
    allow_lists: bool = True,
    verbose: bool = False,
) -> Union[str, List[str]]:
    """
    Adds shape calculations to a vector such as area and perimeter.
    Can also add compactness measurements.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource, List[str, ogr.DataSource]]
        Vector layer(s) or path(s) to vector layer(s).
    
    shapes : Optional[List[str]], optional
        The shapes to calculate. The following a possible:
            * Area          (In same unit as projection)
            * Perimeter     (In same unit as projection)
            * IPQ           (0-1) given as (4*Pi*Area)/(Perimeter ** 2)
            * Hull Area     (The area of the convex hull. Same unit as projection)
            * Compactness   (0-1) given as sqrt((area / hull_area) * ipq)
            * Centroid      (Coordinate of X and Y)
        Default: all shapes.
    
    prefix : str, optional
        Prefix to add to the field names. Default: "".

    allow_lists : bool, optional
        If True, will accept a list of vectors. If False, will raise an error if a list is passed. Default: True.

    verbose : bool, optional
        If True, will print progress. Default: False.

    Returns
    -------
    out_path : str
        Path to the output vector.
    """
    utils_base._type_check(vector, [str, ogr.DataSource, [str, ogr.DataSource]], "vector")
    utils_base._type_check(shapes, [[str], None], "shapes")

    if not allow_lists and isinstance(vector, list):
        raise ValueError("Lists of vectors are not supported when allow_list is False.")

    vector_list = utils_base._get_variable_as_list(vector)
    output = utils_gdal._get_path_from_dataset_list(vector_list)

    for in_vector in vector_list:
        output.append(_vector_add_shapes_in_place(
            in_vector,
            shapes=shapes,
            prefix=prefix,
            verbose=verbose,
        ))

    if isinstance(vector, list):
        return output

    return output[0]
