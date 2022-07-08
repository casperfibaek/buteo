import os
from glob import glob

builds_glob = "~/anaconda3/conda-bld/*.tar.bz2"

for build in glob(builds_glob):
    os.remove(build)
    print(f"Removed {build}")
