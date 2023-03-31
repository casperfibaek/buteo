# Buteo - Geospatial Analysis Meets AI

Buteo is a toolbox designed to simplify the process of working with geospatial data in machine learning. It includes tools for reading, writing, and processing geospatial data, as well as tools for creating labels from vector data and generating patches from geospatial data. Buteo makes it easy to ingest data, create training data, and perform inference on geospatial data.

Please note that Buteo is under active development, and its API may not be entirely stable. Feel free to report any bugs or suggest improvements.

For documentation, visit: https://casperfibaek.github.io/buteo/

**Dependencies** </br>
`numba` (https://numba.pydata.org/) </br>
`gdal` (https://gdal.org/) </br>

**Installation** </br>
Using pip:
```
pip install buteo
```
Using conda:
```
conda install buteo --channel casperfibaek
```

**Quickstart**
```python
import buteo as beo

OUTDIR = "path/to/output/dir"

# Reproject (and other functions) to references. (Vector and raster)
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

```python
import buteo as beo

# Align, stack, and make patches from rasters

SRCDIR = "path/to/src/dir/"

paths_to_aligned_rasters_in_memory = beo.align_rasters(
    SRCDIR + "*.tif:glob",
)

stacked_numpy_arrays = beo.raster_to_array(
    paths_to_aligned_rasters_in_memory,
)

paths_to_patches_in_memory = beo.get_patches(
    path_to_stacked_numpy_arrays,
    256,
    offsets=3,
)

# patches_nr, height, width, channels
paths_to_patches_in_memory
>>> np.ndarray([10000, 256, 256, 9])
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

# Build steps
python -m run_tests; python -m build_documentation;
python -m build; python -m twine upload dist/*;

python -m run_tests && python -m build_documentation
python -m build && python -m twine upload dist/*

python -m build_anaconda -forge -clean;