"""
Build the anaconda package
"""
import os
import sys
from glob import glob
import yaml


# Constants
PLATFORMS = [
    "noarch",
    # "osx-64",
    # "osx-arm64",
    # "linux-64",
    # "win-64",
]
PYTHON = [
    "3.7",
    "3.8",
    "3.9",
    "3.10",
]

with open("./meta.yaml", "r", encoding="latin1") as stream:
    try:
        meta = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

VERSION = meta["package"]["version"]

# Set up the build folder
home = os.path.expanduser("~")
builds_folder = os.path.abspath(home + "/anaconda3/conda-bld/")
builds_glob = os.path.join(builds_folder, "**/*.tar.bz2")

update_only = False
forge = False
clean = False

if "-update_only" in sys.argv:
    update_only = True

if "-clean" in sys.argv:
    clean = True

if "-forge" in sys.argv:
    forge = True

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

# Rename pyproject.toml to ensure it is not used (Why would this be necessary?..)
os.rename("./pyproject.toml", "./_pyproject.toml")

# Build
python_str = f"conda build {os. getcwd()} --py {' --py '.join(PYTHON)}"
if forge:
    python_str = python_str + " -c conda-forge"

os.system(python_str)

# Rename pyproject back to its original name.
os.rename("./_pyproject.toml", "./pyproject.toml")

# Convert to other platforms
for build in glob(builds_glob):
    for platform in PLATFORMS:
        convert_call = f"conda convert {build} --platform={platform} --output-dir={builds_folder}"
        os.system(convert_call)

# Upload to anaconda
for build in glob(builds_glob):
    upload_call = f"anaconda upload {build}"
    os.system(upload_call)

# Build steps
# python -m run_tests && python -m build_documentation
# python -m build && python -m twine upload dist/*
# python -m build_anaconda -forge -clean;
