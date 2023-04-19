"""
Build the anaconda package
"""
import os
import shutil
from glob import glob

# Clean documentation folder
shutil.rmtree('./docs/', ignore_errors=True)

os.mkdir("./docs/")
os.mkdir("./docs/buteo/")

# Build documentation
os.system("pdoc ./buteo -o ./docs")
