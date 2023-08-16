"""
# Geospatial Analysis Meets AI

Buteo is a toolbox designed to simplify the process of working with geospatial data in machine learning. It includes tools for reading, writing, and processing geospatial data, as well as tools for creating labels from vector data and generating patches from geospatial data. Buteo makes it easy to ingest data, create training data, and perform inference on geospatial data.

Please note that Buteo is under active development, and its API may not be entirely stable.
When using, please pin the version of Buteo you are using to avoid breaking changes.
Feel free to report any bugs or suggest improvements.

Author: *Casper Fibaek* (github.com/casperfibaek) </br>

**Dependencies** </br>
`numba` (https://numba.pydata.org/) </br>
`gdal` (https://gdal.org/) </br>

**Installation** </br>
Using pip:
```
pip install gdal
pip install buteo
```
Using conda:
```
conda install gdal
pip install buteo
```

**Quickstart**

### Reproject (and other functions) to references. (Vector and raster)
```python
import buteo as beo

OUTDIR = "path/to/output/dir"

vector_file_correct_projection = "path/to/vector/file.gpkg"
raster_files_wrong_projection = "path/to/raster/files/*.tif:glob"

paths_to_reprojected_rasters = beo.reproject_raster(
    raster_files_with_wrong_projection,
    vector_file_with_correct_projection,
    out_path=outdir
)

paths_to_reprojected_rasters
>>> [path/to/output/dir/file1.tif, path/to/output/dir/file2.tif, ...]
```

### Align, stack, and make patches from rasters
```python
import buteo as beo

SRCDIR = "path/to/src/dir/"

paths_to_aligned_rasters_in_memory = beo.align_rasters(
    SRCDIR + "*.tif:glob",
)

stacked_numpy_arrays = beo.raster_to_array(
    paths_to_aligned_rasters_in_memory,
)

patches = beo.array_to_patches(
    stacked_numpy_arrays,
    256,
    n_offsets=1, # 1 overlap at 1/2 patch size (128)
)

# patches_nr, height, width, channels
patches
>>> np.ndarray([10000, 256, 256, 9])
```

### Predict a raster using a model
```python
import buteo as beo

RASTER_PATH = "path/to/raster/raster.tif"
RASTER_OUT_PATH = "path/to/raster/raster_pred.tif"

array = beo.raster_to_array(RASTER_PATH)

callback = model.predict # from pytorch, keras, etc..

# Predict the raster using overlaps, and borders.
# Merge using different methods. (median, mad, mean, mode, ...)
predicted = predict_array(
    array,
    callback,
    tile_size=256,
)

# Write the predicted raster to disk
beo.array_to_raster(
    predicted,
    reference=RASTER_PATH,
    out_path=RASTER_OUT_PATH,
)
# Path to the predicted raster
>>> "path/to/raster/raster_pred.tif"
```

</br>

| Example Colabs                        |                                                                                                                                                                                                               |
|---------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Create labels from OpenStreetMap data | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/casperfibaek/buteo/blob/master/examples/create_labels_from_osm_data.ipynb)            |
| Scheduled cleaning of geospatial data | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/casperfibaek/buteo/blob/master/examples/rocket_example.ipynb)                         |
| Clip and remove noise from rasters    | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/casperfibaek/buteo/blob/master/examples/clip_and_remove_noise_raster.ipynb)           |
| Sharpen nightlights data              | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/casperfibaek/buteo/blob/master/examples/process_nightlights.ipynb)                    |
| Filters and morphological operations  | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/casperfibaek/buteo/blob/master/examples/filters_and_morphology.ipynb)                 |


</br>

The toolbox is being developed by ESA-Philab, NIRAS, and Aalborg University.

# Dependencies
gdal
numba

optional:
orfeo-toolbox
esa-snap
"""
from .utils import *
from .raster import *
from .vector import *
from .array import *
from .eo import *
from .ai import *

try:
    from osgeo import gdal
    gdal.UseExceptions()
except:
    print("GDAL not installed. Some functions may not work.")

__version__ = "0.9.40"