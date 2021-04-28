# type: ignore
 
from os import terminal_size

# phase_cross_correlatio
# https://github.com/scikit-image/scikit-image/blob/main/skimage/registration/_phase_cross_correlation.py#L109-L276
def mosaic_sentinel1(
    folder,
    output_folder,
    tmp_folder,
    interest_area,
    target_projection,
    polarisation="vv",
):
    1.  save all metadatas from images
    2.  find sentinel 2 tiles that intersect interest area
    3.  for tile in sentinel2_tiles:
            test does extent intersect tile
            clip to tile and reproject to tmp_folder

            base = the most central date and overlap > 50

            coregister images to base

            order images by time distance to base

            3D elipsodial weighted median filter

            save to output_folder
        
    4.  update and coregister all tiles
    5.  orfeo-toolbox mosaic tiles.
