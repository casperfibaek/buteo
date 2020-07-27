from lib.orfeo_toolbox import meanshift_segmentation
from lib.utils_core import step_function


test_grid = {
    'spatialr': [3, 5, 7, 9],
    'ranger': [0.2, 0.3],
    'thres': [0.1, 0.25, 0.5],
    'minsize': [50, 100, 200]
}

step_function(
    meanshift_segmentation,
    'C:\\Users\\CFI\\Desktop\\segmentation_test\\b4_downtown.vrt',
    # 'C:\\Users\\CFI\\Desktop\\segmentation_test\\shp\\bob.shp',
    grid=test_grid,
    outfile=True,
    outfile_arg=1,
    outfile_prefix='C:\\Users\\CFI\\Desktop\\segmentation_test\\shp\\meanshift_',
    outfile_suffix='.shp',
)

