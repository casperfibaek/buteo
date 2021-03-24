def intersection_rasters(raster_1, raster_2):
    raster_1 = raster_1 if isinstance(raster_1, gdal.Dataset) else gdal.Open(raster_1)
    raster_2 = raster_2 if isinstance(raster_2, gdal.Dataset) else gdal.Open(raster_2)

    img_1 = raster_to_metadata(raster_1)
    img_2 = raster_to_metadata(raster_2)

    driver = ogr.GetDriverByName('Memory')
    dst_source = driver.CreateDataSource('clipped_rasters')
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(4326)
    dst_layer = dst_source.CreateLayer('unused', dst_srs, geom_type=ogr.wkbPolygon)

    geom1 = gdal.OpenEx(img_1['footprint'])
    layer1 = geom1.GetLayer()
    feature1 = layer1.GetFeature(0)
    feature1_geom = feature1.GetGeometryRef()

    geom2 = gdal.OpenEx(img_2['footprint'])
    layer2 = geom2.GetLayer()
    feature2 = layer2.GetFeature(0)
    feature2_geom = feature2.GetGeometryRef()

    if feature2_geom.Intersects(feature1_geom):
        intersection = feature2_geom.Intersection(feature1_geom)
        dstfeature = ogr.Feature(dst_layer.GetLayerDefn())
        dstfeature.SetGeometry(intersection)
        dst_layer.CreateFeature(dstfeature)
        dstfeature.Destroy()
        
        return dst_source
    else:
        return False