"""Fixtures for utilities tests."""

import pytest
import os
import tempfile
import numpy as np
from pathlib import Path

@pytest.fixture
def temp_directory():
    """Create a temporary directory for test files.
    
    Returns:
        str: Path to a temporary directory
    """
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Clean up the directory after tests
    for root, dirs, files in os.walk(temp_dir, topdown=False):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            os.rmdir(os.path.join(root, dir))
    os.rmdir(temp_dir)

@pytest.fixture
def sample_bbox():
    """Create a sample bounding box for testing.
    
    Returns:
        list: A bounding box in the format [xmin, xmax, ymin, ymax]
    """
    return [0.0, 1.0, 0.0, 1.0]

@pytest.fixture
def complex_bbox():
    """Create a more complex bounding box for testing.
    
    Returns:
        list: A bounding box in the format [xmin, xmax, ymin, ymax]
    """
    return [-122.5, -122.3, 37.7, 37.8]

@pytest.fixture
def sample_file_path(temp_directory):
    """Create a sample file path for testing.
    
    Returns:
        str: Path to a temporary file
    """
    return os.path.join(temp_directory, "test_file.txt")

@pytest.fixture
def sample_directory_structure(temp_directory):
    """Create a sample directory structure for testing.
    
    Returns:
        str: Path to the root of the directory structure
    """
    # Create subdirectories
    subdir1 = os.path.join(temp_directory, "subdir1")
    subdir2 = os.path.join(temp_directory, "subdir2")
    
    os.makedirs(subdir1, exist_ok=True)
    os.makedirs(subdir2, exist_ok=True)
    
    # Create some files
    with open(os.path.join(temp_directory, "file1.txt"), "w") as f:
        f.write("Test file 1")
    
    with open(os.path.join(subdir1, "file2.txt"), "w") as f:
        f.write("Test file 2")
    
    with open(os.path.join(subdir2, "file3.txt"), "w") as f:
        f.write("Test file 3")
    
    return temp_directory
