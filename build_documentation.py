"""
Build the anaconda package
"""
import os
import shutil

# Clean documentation folder
shutil.rmtree('./docs/', ignore_errors=True)

os.mkdir("./docs/")
os.mkdir("./docs/buteo/")

# Build documentation
os.system("pdoc ./buteo -o ./docs --docformat google")
