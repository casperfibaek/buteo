"""### Combined vector geometry conversion functionality. ###"""

# Standard library
from typing import Union, Optional

# External
from osgeo import ogr

# Internal
from buteo.utils import (
    utils_base,
    utils_io,
    utils_gdal,
    utils_path,
)
from buteo.core_vector.core_vector_write import vector_create_copy
from buteo.core_vector.conversion.multipart import (
    check_vector_is_multipart,
    vector_multipart_to_singlepart,
    vector_singlepart_to_multipart,
)
from buteo.core_vector.conversion.multitype import (
    vector_change_multitype,
)
from buteo.core_vector.conversion.dimensionality import (
    vector_change_dimensionality,
)


def vector_convert_geometry(
    vector: Union[str, ogr.DataSource],
    *,
    multitype: Optional[bool] = None,
    multipart: Optional[bool] = None,
    z: Optional[bool] = None,
    m: Optional[bool] = None,
    output_path: Optional[str] = None,
    layer_name_or_id: Union[str, int] = 0,
    z_attribute: Optional[str] = None,
    m_attribute: Optional[str] = None,
    prefix: str = "",
    suffix: str = "",
    overwrite: bool = False,
) -> str:
    """ Convert the geometry of a vector to a different subtype.

    Convert between multiparts and singleparts, 2D and 3D, and with or without M values.

    Parameters
    ----------
    vector : str or ogr.DataSource
        The vector to convert.
    multitype : bool, optional
        If True, the output vector will be of the "multi"types. That is MultiPolygon, MultiPoint, etc. Default: None
    multipart : bool, optional
        If True, the output vector will be multiparts. Features will be merged into a single feature with multiple geometries.
        The merged features will have the same attributes.
        If False, the output will be singleparts. If a multi feature is encountered, it will be split into multiple features.
        The split features will have the same attributes.
        If None, no changes. Default: None
    z : bool, optional
        If True, the output vector will be 3D. If None, no changes. Default: None
    m : bool, optional
        If True, the output vector will have M (measure) values. If None, no changes. Default: None
    output_path : str, optional
        The output path. Default: None (in-memory is created)
    layer_name_or_id : str or int, optional
        The name or index of the layer to convert. Default: 0
    z_attribute : str, optional
        The name of the attribute to use for Z values. If None, 0.0 is inserted Default: None
    m_attribute : str, optional
        The name of the attribute to use for M values. If None, 0.0 is inserted Default: None
    prefix : str, optional
        Prefix to add to output path. Default: ""
    suffix : str, optional
        Suffix to add to output path. Default: ""
    overwrite : bool, optional
        If True, overwrites existing files. Default: False

    Returns
    -------
    str
        The path to the converted vector
    """
    utils_base._type_check(vector, [str, ogr.DataSource], "vector")
    utils_base._type_check(multitype, [type(None), bool], "multitype")
    utils_base._type_check(multipart, [type(None), bool], "multipart")
    utils_base._type_check(z, [type(None), bool], "z")
    utils_base._type_check(m, [type(None), bool], "m")
    utils_base._type_check(output_path, [type(None), str], "output_path")
    utils_base._type_check(layer_name_or_id, [str, int], "layer_name_or_id")
    utils_base._type_check(z_attribute, [type(None), str], "z_attribute")
    utils_base._type_check(m_attribute, [type(None), str], "m_attribute")
    utils_base._type_check(prefix, [str], "prefix")
    utils_base._type_check(suffix, [str], "suffix")
    utils_base._type_check(overwrite, [bool], "overwrite")

    if multitype is False and multipart is True:
        raise ValueError("Cannot set both multitype to False and multipart to True.")

    in_path = utils_io._get_input_paths(vector, "vector")
    out_path = utils_io._get_output_paths(in_path, output_path, prefix=prefix, suffix=suffix) # type: ignore

    utils_io._check_overwrite_policy(out_path, overwrite)
    utils_io._delete_if_required_list(out_path, overwrite)

    in_path = in_path[0]
    out_path = out_path[0]

    # We don't check the other way, because not all the features are possibly multiparts
    if not check_vector_is_multipart(in_path, layer_name_or_id) and not multipart:
        multipart = None

    temp_path_1 = utils_path._get_temp_filepath("temp_path_1.fgb", add_timestamp=True, add_uuid=True)
    temp_path_2 = utils_path._get_temp_filepath("temp_path_2.fgb", add_timestamp=True, add_uuid=True)
    temp_path_3 = utils_path._get_temp_filepath("temp_path_3.fgb", add_timestamp=True, add_uuid=True)

    # First we convert the types:
    converted = in_path
    if multipart is True:
        converted = vector_singlepart_to_multipart(
            converted,
            layer_name_or_id=layer_name_or_id,
            output_path=temp_path_1,
            overwrite=overwrite,
        )
    elif multipart is False:
        converted = vector_multipart_to_singlepart(
            converted,
            layer_name_or_id=layer_name_or_id,
            output_path=temp_path_1,
            overwrite=overwrite,
        )

    # Then we convert the multitype
    if multitype is not None:
        converted = vector_change_multitype(
            converted,
            multitype,
            layer_name_or_id=layer_name_or_id,
            output_path=temp_path_2,
            overwrite=overwrite,
        )

    # Then we convert the dimensionality
    if z is not None or m is not None:
        converted = vector_change_dimensionality(
            converted,
            z=z,
            m=m,
            layer_name_or_id=layer_name_or_id,
            output_path=temp_path_3,
            z_attribute=z_attribute,
            m_attribute=m_attribute,
            overwrite=overwrite,
        )

    # Copy the final result to the output path
    output = vector_create_copy(
        converted,
        out_path=out_path,
        overwrite=overwrite,
    )

    utils_gdal.delete_dataset_if_in_memory_list([temp_path_1, temp_path_2, temp_path_3])

    if not isinstance(output, str):
        raise ValueError("Could not create the output vector.")

    return output
