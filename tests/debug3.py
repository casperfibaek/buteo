import sys; sys.path.append("../")
import buteo as beo

path_s2 = "./features/japan.tif"
path_kg = "./features/kg.tif"
 
bbox_ltlng = beo.raster_to_metadata(path_s2)['bbox_latlng']
bbox_vector = beo.vector_from_bbox(bbox_ltlng, projection="EPSG:4326")
bbox_vector_buffered = beo.vector_buffer(bbox_vector, distance=0.1)

kg_clipped = beo.raster_clip(path_kg, bbox_vector_buffered, to_extent=True, adjust_bbox=False)
kg_aligned = beo.raster_align(kg_clipped, reference=path_s2, method='reference', resample_alg='nearest', out_path='./features/kg_aligned.tif')
