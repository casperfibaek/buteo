# type: ignore
import sys

sys.path.append("..")
sys.path.append("../../")

from glob import glob
from buteo.raster.io import internal_raster_to_metadata
from buteo.gdal_utils import parse_projection

# phase_cross_correlatio
# https://github.com/scikit-image/scikit-image/blob/main/skimage/registration/_phase_cross_correlation.py#L109-L276
def mosaic_sentinel1(
    folder, output_folder, tmp_folder, interest_area=None, target_projection=None,
):
    metadatas = []
    images = glob(folder + "*Gamma0*.tif")
    used_projections = []
    extent = []

    for idx, image in enumerate(images):
        metadata = internal_raster_to_metadata(image, create_geometry=True)
        used_projections.append(metadata["projection"])

        x_min, x_max, y_min, y_max = metadata["extent_ogr_latlng"]

        if idx == 0:
            extent = metadata["extent_ogr_latlng"]
        else:
            if x_min < extent[0]:
                extent[0] = x_min
            if x_max > extent[1]:
                extent[1] = x_max
            if y_min < extent[2]:
                extent[2] = y_min
            if y_max > extent[3]:
                extent[3] = y_max

        metadatas.append(metadata)

    use_projection = None
    if target_projection is None:
        projection_counter: dict = {}
        for proj in used_projections:
            if proj in projection_counter:
                projection_counter[proj] += 1
            else:
                projection_counter[proj] = 1

        # Choose most common projection
        most_common_projection = sorted(
            projection_counter, key=projection_counter.__getitem__, reverse=True
        )
        use_projection = parse_projection(most_common_projection[0])
    else:
        use_projection = parse_projection(target_projection)

    import pdb

    pdb.set_trace()

    # 1.  save all metadatas from images
    # 2.  find sentinel 2 tiles that intersect interest area
    # 3.  for tile in sentinel2_tiles:
    #         test does extent intersect tile
    #         clip to tile and reproject to tmp_folder

    #         base = the most central date and overlap > 50

    #         coregister images to base

    #         order images by time distance to base

    #         3D elipsodial weighted median filter

    #         save to output_folder

    # 4.  update and coregister all tiles
    # 5.  orfeo-toolbox mosaic tiles.


if __name__ == "__main__":
    folder = "C:/Users/caspe/Desktop/paper_3_Transfer_Learning/data/sentinel1/"
    merged = folder + "merged/"
    processed = folder + "mosaic_2020/"
    tmp = folder + "tmp/"

    mosaic_sentinel1(processed, merged, tmp)
