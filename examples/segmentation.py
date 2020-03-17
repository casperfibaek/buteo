import sys; sys.path.append('..')
from lib.orfeo_toolbox import meanshift_segmentation
from lib.utils_core import step_function

folder = '/mnt/c/users/caspe/desktop/Analysis/Phase1/'

test_grid = {
    'spatialr': [3, 5, 7, 9],
    'ranger': [0.2, 0.3, 0.4],
    'thres': [0.1, 0.25, 0.5],
    'minsize': [50, 100, 200]
}

step_function(
    meanshift_segmentation,
    folder + 'phase1_layers.vrt',
    grid=test_grid,
    outfile=True,
    outfile_arg=1,
    outfile_prefix=folder + 'segmentations/segmentation_',
    outfile_suffix='.shp',
)