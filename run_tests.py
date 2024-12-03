""" Run some or all the tests. """
import os
from sys import argv

if not os.path.exists("tests/"):
    raise FileNotFoundError("No tests found.")

RUN_TESTS = "all"
if len(argv) == 1:
    RUN_TESTS = "all"
elif len(argv) == 2:
    RUN_TESTS = argv[1]
else:
    raise ValueError("Unknow test specified")

if RUN_TESTS == "all":
    os.system("python -m pytest tests/")
else:
    os.system(f"python -m pytest tests/{RUN_TESTS}")
