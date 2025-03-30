"""Common test fixtures for all test modules."""

import os
import sys
import pytest
import numpy as np
from osgeo import ogr, osr

# Add the parent directory to sys.path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Common fixtures that may be used across multiple test modules

@pytest.fixture
def common_temp_dir(tmp_path):
    """Create a common temporary directory for test data.
    
    Returns:
        pathlib.Path: Path to the temporary directory.
    """
    return tmp_path
