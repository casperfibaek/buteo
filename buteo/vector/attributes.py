
def vector_get_attribute_table(vector, process_layers="first", geom=False):
    ref = vector_to_reference(vector)
    metadata = vector_to_metadata(ref, process_layers=process_layers)

    dataframes = []

    for vector_layer in range(metadata["layer_count"]):
        attribute_table_header = None
        feature_count = None

        if process_layers != "first":
            attribute_table_header = metadata["layers"][vector_layer]["field_names"]
            feature_count = metadata["layers"][vector_layer]["feature_count"]
        else:
            attribute_table_header = metadata["field_names"]
            feature_count = metadata["feature_count"]

        attribute_table = []

        layer = ref.GetLayer(vector_layer)

        for _ in range(feature_count):
            feature = layer.GetNextFeature()
            attributes = [feature.GetFID()]

            for field_name in attribute_table_header:
                attributes.append(feature.GetField(field_name))

            if geom:
                geom_defn = feature.GetGeometryRef()
                attributes.append(geom_defn.ExportToIsoWkt())
            
            attribute_table.append(attributes)

        attribute_table_header.insert(0, "fid")

        if geom:
            attribute_table_header.append("geom")
        
        df = pd.DataFrame(attribute_table, columns=attribute_table_header)

        if process_layers == "first": return df
        
        dataframes.append(df)

    return dataframes
