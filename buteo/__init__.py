"""
# Buteo - Facilitating EO-Driven Decision Support Systems

The Buteo-Toolbox is a series of modules that ease the creation of Earth Observation Driven Spatial Decision Support Systems. The modules are located in the lib folder, geometry used for testing and clipping is located in geometry. In the examples folder there are jupyter notebooks that showcase analysis' done using the toolbox.

Documentation available at: https://casperfibaek.github.io/buteo/

**Dependencies** </br>
`numpy` (https://numpy.org/) </br>
`gdal` (https://gdal.org/) </br>

**Installation** </br>
`pip install buteo` </br>
`conda install buteo --channel casperfibaek` </br>

**Quickstart**
```python
import buteo as beo

vector_file_correct_projection = "path/to/vector/file.gpkg"
raster_files_wrong_projection = "path/to/raster/files/*.tif:glob"

outdir = 'path/to/output/dir'

paths_to_reprojected_rasters = beo.reproject_raster(
    raster_files_with_wrong_projection,
    vector_file_with_correct_projection,
    out_path=outdir
)

paths_to_reprojected_rasters
>>> [path/to/output/dir/file1.tif, path/to/output/dir/file2.tif, ...]
```
</br>

| Example Colabs                        |                                                                                                                                                                                                               |
|---------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Create labels from OpenStreetMap data | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/casperfibaek/buteo/blob/master/examples/create_labels_from_osm_data.ipynb)            |
| Scheduled cleaning of geospatial data | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/casperfibaek/buteo/blob/master/examples/rocket_example.ipynb)                         |
| Clip and remove noise from rasters    | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/casperfibaek/buteo/blob/master/examples/clip_and_remove_noise_raster.ipynb)           |
| Filters and morphological operations  | [![Open All Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/casperfibaek/buteo/blob/master/examples/filters_and_morphology.ipynb)                 |

</br>

# Modules:

## raster

- read and verify a raster or a list of rasters
- clip rasters to other rasters or vectors
- align a list of rasters for analysis
- shift, resample and reproject raster data
- easily manage nodata values
- parallel zonal statistics (link - host)

## vector

- read and verify integrity
- parallel zonal statistics (link)
- clip, buffer

## filter

- custom convolution based filters.
- global filters and multispectral indices (link - host)
- textures for analysis
- noise reduction of SAR imagery (link - host)
- kernel designer & reduction bootcamp

## terrain

- download srtm, aster and the danish national DEM
- basic propocessing of the DEM's.

## earth_observation

- download sentinel 1, 2, 3, 5, landsat and modis data
- process sentinel 1 and 2 (sentinel 1 requires esa-snap dep.)
- generate mosaics of sentinel 1 and 2
- pansharpen bands
- noise reduction of SAR imagery (link)
- multispectral indices (link)

## machine_learning

- patch extraction of tiles and geometries, allows overlaps, for CNN's
- machine learning utilities library: rotate images, add noise etc..
- model for building extraction for sentinel 1 and 2
- model for pansharpening sentinel 2

## extra

- custom orfeo-toolbox python bindings
- ESA snap GPT python bindings and graphs

The system is under active development and is not ready for public release. It is being developed by ESA, NIRAS, and Aalborg University.

# Dependencies

gdal
numba

optional:
orfeo-toolbox
esa-snap

# Todo

- finish filters library - kernel tests
- update zonal statistics & parallelise vector math
- remove dependencies: sen1mosaic
- create models for pansharpening, buildings and noise-reduction
- generate examples
- synthetic sentinel 2?
- Move deep learning / machine learning stuff in to seperate package (BUTEO & BUTEO_DL)
- Add checks modules: raster_overlaps, all_rasters_intersect, etc...

# Functions todo

raster_footprints
raster_mosaic
raster_proximity
raster_hydrology
raster_vectorize

vector_grid
vector_select
vector_buffer_etc..

machine_learning_extract_sample_points

python -m run_tests; python -m build_documentation;
python -m build; python -m twine upload dist/*;
python -m build_anaconda -forge -clean;
```
"""

from .utils import *
from .raster import *
from .vector import *
