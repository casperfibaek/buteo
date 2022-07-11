"""
Build the anaconda package

TODO:
    - Make the process parallel using threads. Should speed build times up significantly.
"""
import os
import sys
from glob import glob

import regex

# Constants
PLATFORMS = [
    # "osx-64",
    # "osx-arm64",
    "linux-64",
    # "win-64",
]
PYTHON = [
    # "3.7",
    # "3.8",
    "3.9",
    # "3.10",
]


# Set up the build folder
home = os.path.expanduser("~")
builds_folder = os.path.abspath(home + "/anaconda3/conda-bld/")
builds_glob = os.path.join(builds_folder, "**/*.tar.bz2")

# Find version
found = False
update_only = False
forge = False
clean = False
for arg in sys.argv:
    if "--version=" in arg:
        VERSION = arg.split("=")[1]
        if len(VERSION.split(".")) != 3 or not regex.match(r'^\d+(\.\d+)*$', VERSION):
            print("Version must be in the format x.y.z")
            sys.exit(1)

        found = True

    if "-update_only" in arg:
        update_only = True

    if "-clean" in arg:
        clean = True
    
    if "-forge" in arg:
        forge = True

if found is False:
    raise Exception("Version not found. Please specify with --version=<version>")

# Update setup.py
with open('./setup.py', 'r', encoding="utf-8") as setup_file :
    setup_file_data = setup_file.read()

all_lines_setup = setup_file_data.splitlines()

for idx, val in enumerate(all_lines_setup):
    if "VERSION = " in val:
        setup_file_data = setup_file_data.replace(val, f'VERSION = "{VERSION}"')
        break

# Write the file out again
with open('./setup.py', 'w', encoding="utf-8") as file:
    file.write(setup_file_data)


# Update meta.yaml
with open('./meta.yaml', 'r', encoding="utf-8") as meta_file :
    meta_file_data = meta_file.read()

all_lines_meta = meta_file_data.splitlines()

for idx, val in enumerate(all_lines_meta):
    if "version: " in val:
        meta_file_data = meta_file_data.replace(val, f'  version: "{VERSION}"')

    if "git_rev: " in val:
        meta_file_data = meta_file_data.replace(val, f'  git_rev: "{VERSION}"')

# Write the file out again
with open('./meta.yaml', 'w', encoding="utf-8") as file:
    file.write(meta_file_data)

if update_only:
    print("Updating only")
    sys.exit(0)

# Clean previous builds
if clean:
    print("Cleaning build folder of previous builds")

    os.system("conda build purge")

    for build in glob(builds_glob):
        os.remove(build)
        print(f"Removed {build}")

# Build
python_str = f"conda build {os. getcwd()} --py {' --py '.join(PYTHON)}"
if forge:
    python_str = python_str + " -c conda-forge"

os.system(python_str)

# Convert to other platforms
for build in glob(builds_glob):
    for platform in PLATFORMS:
        convert_call = f"conda convert {build} --platform={platform} --output-dir={builds_folder}"
        os.system(convert_call)

# Upload to anaconda
for build in glob(builds_glob):
    upload_call = f"anaconda upload {build}"
    import pdb; pdb.set_trace()
    os.system(upload_call)

# python build.py --version=0.1.0 -clean
