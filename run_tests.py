""" Run some or all the tests. """
import os
import sys

if not os.path.exists("tests/"):
    raise FileNotFoundError("No tests found.")

if len(sys.argv) == 1:
    # Run all tests
    os.system("python -m pytest tests/")
elif len(sys.argv) == 2:
    # Run a specific test module or folder
    test_path = sys.argv[1]
    if test_path == "all":
        os.system("python -m pytest tests/")
    else:
        # Check if path is relative to tests/ or absolute
        test_path = test_path if test_path.startswith("tests/") else f"tests/{test_path}"
        os.system(f"python -m pytest {test_path}")
else:
    # Run multiple specific test files
    test_paths = sys.argv[1:]
    # Create a space-separated list of test paths
    test_paths_string = " ".join(test_paths)
    os.system(f"python -m pytest {test_paths_string}")
