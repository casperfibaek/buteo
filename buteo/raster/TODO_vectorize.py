# source_raster = gdal.Open(raster_path)      
# band = gdal.Open(raster_path).GetRasterBand(1)
# driver = ogr.GetDriverByName('ESRI Shapefile')        
# out_data = driver.CreateDataSource(shapefile_path)
# # getting projection from source raster
# srs = osr.SpatialReference()
# srs.ImportFromWkt(source_raster.GetProjectionRef())
# # create layer with projection
# out_layer = out_data.CreateLayer(raster_path.split('.')[0], srs)        
# new_field = ogr.FieldDefn('field_name', ogr.OFTReal)
# out_layer.CreateField(new_field)        
# gdal.FPolygonize(band, band, out_layer, 0, [], callback=None)        
# out_data.Destroy()
# source_raster = None