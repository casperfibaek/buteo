import os
from glob import glob

home = os.path.expanduser("~")
builds_folder = os.path.abspath(home + "/anaconda3/conda-bld/")
builds_glob = os.path.join(builds_folder, "**/*.tar.bz2")

for build in glob(builds_glob):
    os.remove(build)
    print(f"Removed {build}")

os.system("conda build . --py 3.7 --py 3.8 --py 3.9 --py 3.10 -c conda-forge")

platforms = ["osx-64","osx-arm64", "linux-32", "linux-64", "win-32", "win-64"]
platforms_str = " ".join(platforms)

for build in glob(builds_glob):
    os.system(f"conda convert --platform f{platforms_str} f{build} -o f{builds_folder}")

for build in glob(builds_glob):
    os.system(f"anaconda upload f{build}")
