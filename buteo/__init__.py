"""
## Welcome to the documentation for the **Buteo** package!

**Dependencies** </br>
`numpy` (https://numpy.org/) </br>
`gdal` (https://gdal.org/) </br>

**Installation** </br>
`pip install buteo` </br>
`conda install buteo --channel casperfibaek` </br>

**Quickstart**
```python
from glob import glob
from buteo.raster import reproject_raster

vector_file_with_correct_projection = 'path/to/vector/file.gpkg'
raster_files_with_wrong_projection = 'path/to/raster/files/*.tif:glob'

outdir = 'path/to/output/dir'

result = reproject_raster(
    raster_files_with_wrong_projection,
    vector_file_with_correct_projection,
    out_path=outdir
)

result
>>> [path/to/output/dir/file1.tif, path/to/output/dir/file2.tif, ...]
```
"""

from .utils import *
from .raster import *
from .vector import *
