"""### Work with vector data and attributes with ease. ###"""

# First import core modules
from buteo.core_vector.core_vector_read import open_vector
from buteo.core_vector.core_vector_info import get_metadata_vector
from buteo.core_vector.core_vector_index import vector_add_index
from buteo.core_vector.core_vector_merge import vector_merge_layers

# Then import the higher level modules
from .clip import *
from .dissolve import *
from .grid import *
from .intersect import *
from .rasterize import *
from .reproject import *
from .buffer import *
from .sample import *
from .extract_by_attribute import *
from .extract_by_location import *
from .zonal_statistics import *
