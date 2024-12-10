def vector_create_index(
    vector: Union[ogr.DataSource, str, List[Union[ogr.DataSource, str]]],
) -> Union[str, List[str]]:
    """Adds a spatial index to the vector in place, if it doesn't have one.

    Parameters
    ----------
    vector : Union[ogr.DataSource, str, List[Union[ogr.DataSource, str]]]
        A vector layer(s) or path(s) to a vector file.

    Returns
    -------
    Union[str, List[str]]
        The path(s) to the input vector file(s).
    """
    utils_base._type_check(vector, [str, ogr.DataSource, [str, ogr.DataSource]], "vector")

    input_is_list = isinstance(vector, list)
    in_paths = utils_io._get_input_paths(vector, "vector")

    try:
        for in_vector in in_paths:
            metadata = _get_basic_metadata_vector(in_vector)
            ref = _vector_open(in_vector)

            for layer in metadata["layers"]:
                name = layer["layer_name"]
                geom = layer["column_geom"]

                sql = f"SELECT CreateSpatialIndex('{name}', '{geom}') WHERE NOT EXISTS (SELECT HasSpatialIndex('{name}', '{geom}'));"
                ref.ExecuteSQL(sql, dialect="SQLITE")
    except:
        raise RuntimeError(f"Error while creating indices for {vector}") from None

    if input_is_list:
        return in_paths

    return in_paths[0]

# vector_create_index
# vector_delete_index
# check_vector_has_valid_index
