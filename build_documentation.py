"""
Build the anaconda package

TODO:
    - Make the process parallel using threads. Should speed build times up significantly.
"""
import os
import shutil
from glob import glob

# Clean documentation folder
shutil.rmtree('./docs/', ignore_errors=True)

# Build documentation
os.system("pdoc3 ./buteo -o ./docs --html -f")

# Move from buteo -> docs
for file in glob("./docs/buteo/*"):
    shutil.move(file, "./docs/")

# Remove buteo subfolder
os.removedirs("./docs/buteo")
