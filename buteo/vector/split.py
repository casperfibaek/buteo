"""### Split functions for vector layers. ###

Dissolve vectors by attributes or geometry.
"""

# Standard library
import os
from typing import Optional, Union
from uuid import uuid4

# External
from osgeo import ogr

# Internal
from buteo.utils import utils_base, utils_path
from buteo.vector import core_vector



def vector_split_by_fid(
    vector: Union[str, ogr.DataSource],
    out_folder: Optional[str] = None,
    output_layer_idx: bool = False,
    prefix: str = "",
    suffix: str = "",
) -> str:
    """Split a vector by feature id.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource]
        Input vector.

    out_folder : Optional[str], optional
        Output folder, default: None.

    prefix : str, optional
        Prefix for output files, default: "".

    suffix : str, optional
        Suffix for output files, default: "".

    Returns
    -------
    str
        Output paths.
    """
    utils_base._type_check(vector, [ogr.DataSource, str], "vector")
    utils_base._type_check(out_folder, [str, None], "out_folder")
    assert utils_path._check_dir_exists(out_folder) or out_folder is None, "out_folder does not exist."

    opened = core_vector.vector_open(vector, writeable=False)
    metadata = core_vector._get_basic_metadata_vector(opened)

    out_paths = []

    for layer_index in range(metadata["layer_count"]):
        layer = opened.GetLayerByIndex(layer_index)
        layer.ResetReading()

        feature = layer.GetNextFeature()
        while feature:
            feature_id = feature.GetFID()

            if out_folder is None:
                out_path = f"/vsimem/{prefix}{str(uuid4().int)}_{layer_index}_{feature_id}{suffix}.gpkg"
            else:
                if output_layer_idx:
                    out_path = os.path.join(
                        out_folder,
                        f"{prefix}{layer_index}_{feature_id}{suffix}.gpkg",
                    )
                else:
                    out_path = os.path.join(
                        out_folder,
                        f"{prefix}{feature_id}{suffix}.gpkg",
                    )

            out_driver = ogr.GetDriverByName("GPKG")
            out_ds = out_driver.CreateDataSource(out_path)
            out_layer = out_ds.CreateLayer(
                layer.GetName(),
                srs=layer.GetSpatialRef(),
                geom_type=layer.GetGeomType(),
            )

            out_layer.CreateFeature(feature)
            out_layer.SyncToDisk()

            out_ds.FlushCache()
            out_ds, out_layer, feature = (None, None, None)

            out_paths.append(out_path)

            feature = layer.GetNextFeature()

    return out_paths


def vector_split(
    vector: Union[str, ogr.DataSource],
    out_folder: Optional[str] = None,
    output_layer_idx: bool = False,
    prefix: str = "",
    suffix: str = "",
) -> str:
    """Split a vector by feature id.

    Parameters
    ----------
    vector : Union[str, ogr.DataSource]
        Input vector.

    out_folder : Optional[str], optional
        Output folder, default: None.

    prefix : str, optional
        Prefix for output files, default: "".

    suffix : str, optional
        Suffix for output files, default: "".

    Returns
    -------
    str
        Output paths.
    """
    utils_base._type_check(vector, [ogr.DataSource, str], "vector")
    utils_base._type_check(out_folder, [str, None], "out_folder")
    assert utils_path._check_dir_exists(out_folder) or out_folder is None, "out_folder does not exist."

    opened = core_vector.vector_open(vector, writeable=False)
    metadata = core_vector._get_basic_metadata_vector(opened)

    out_paths = []

    for layer_index in range(metadata["layer_count"]):
        layer = opened.GetLayerByIndex(layer_index)
        layer.ResetReading()

        feature = layer.GetNextFeature()
        feature_id = 0
        while feature:

            if out_folder is None:
                out_path = f"/vsimem/{prefix}{str(uuid4().int)}_{layer_index}_{feature_id}{suffix}.gpkg"
            else:
                if output_layer_idx:
                    out_path = os.path.join(
                        out_folder,
                        f"{prefix}{layer_index}_{feature_id}{suffix}.gpkg",
                    )
                else:
                    out_path = os.path.join(
                        out_folder,
                        f"{prefix}{feature_id}{suffix}.gpkg",
                    )

            out_driver = ogr.GetDriverByName("GPKG")
            out_ds = out_driver.CreateDataSource(out_path)
            out_layer = out_ds.CreateLayer(
                layer.GetName(),
                srs=layer.GetSpatialRef(),
                geom_type=layer.GetGeomType(),
            )

            out_layer.CreateFeature(feature)
            out_layer.SyncToDisk()

            out_ds.FlushCache()
            out_ds, out_layer, feature = (None, None, None)

            out_paths.append(out_path)

            feature = layer.GetNextFeature()
            feature_id += 1

    return out_paths
