import os
from glob import glob

home = os.path.expanduser("~")
builds_folder = os.path.abspath(home + "/anaconda3/conda-bld/")
builds_glob = os.path.join(builds_folder, "**/*.tar.bz2")


for build in glob(builds_glob):
    os.remove(build)
    print(f"Removed {build}")
