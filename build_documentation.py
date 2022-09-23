"""
Build the anaconda package
"""
import os
import shutil
from glob import glob

# Clean documentation folder
shutil.rmtree('./docs/', ignore_errors=True)

os.mkdir("./docs/")

# Build documentation
os.system("pdoc3 ./buteo -o ./docs --html -f")

# Move from buteo -> docs
for file in glob("./docs/buteo/*"):
    shutil.move(file, "./docs/")

# Remove buteo subfolder
os.removedirs("./docs/buteo")
