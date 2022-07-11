""" Build the anaconda package """
import os
import sys
from glob import glob


# Constants
PLATFORMS = ["osx-64","osx-arm64", "linux-64", "win-64"]
PYTHON = ["3.7", "3.8", "3.9", "3.10"]

python_str = f"conda build . --py {' --py '.join(PYTHON)} -c conda-forge"
platforms_str = " ".join(PLATFORMS)

# Set up the build folder
home = os.path.expanduser("~")
builds_folder = os.path.abspath(home + "/anaconda3/conda-bld/")
builds_glob = os.path.join(builds_folder, "**/*.tar.bz2")

# Find version
found = False
update_only = False
for arg in sys.argv:
    if "--version=" in arg:
        VERSION = arg.split("=")[1]
        found = True

    if "-u" in arg:
        update_only = True

if found is False:
    raise Exception("Version not found. Please specify with --version=<version>")

# Update setup.py
with open('./setup.py', 'r') as setup_file :
    setup_file_data = setup_file.read()

all_lines_setup = setup_file_data.splitlines()

for idx, val in enumerate(all_lines_setup):
    if "VERSION = " in val:
        setup_file_data = setup_file_data.replace(val, f'VERSION = "{VERSION}"')
        break

# Write the file out again
with open('./setup.py', 'w') as file:
    file.write(setup_file_data)


# Update meta.yaml
with open('./meta.yaml', 'r') as meta_file :
    meta_file_data = meta_file.read()

all_lines_meta = meta_file_data.splitlines()

for idx, val in enumerate(all_lines_meta):
    if "version: " in val:
        meta_file_data = meta_file_data.replace(val, f'  version: "{VERSION}"')

    if "git_rev: " in val:
        meta_file_data = meta_file_data.replace(val, f'  git_rev: "{VERSION}"')

# Write the file out again
with open('./meta.yaml', 'w') as file:
    file.write(meta_file_data)

if update_only:
    print("Updating only")
    sys.exit(0)

# Build
for build in glob(builds_glob):
    os.remove(build)
    print(f"Removed {build}")

os.system(python_str)

for build in glob(builds_glob):
    os.system(f"conda convert --platform f{platforms_str} f{build} -o f{builds_folder}")

for build in glob(builds_glob):
    os.system(f"anaconda upload f{build}")
