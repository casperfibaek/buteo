"""
This modules uses the sen2cor tool to generate the L2A products.

TODO:
    - Create versions for Windows and MacOS
"""

import sys; sys.path.append("../../") # Path: buteo/earth_observation/s2_l2_process.py
import os
from glob import glob
from shutil import rmtree

from buteo.earth_observation.s2_utils import unzip_files_to_folder


def l2_process(
    l1_dir,
    tmp_dir,
    dst_dir,
    gip="default",
    resolution=10,
    sen2cor_path="/home/cfi/Desktop/Sen2Cor-02.09.00-Linux64/bin/L2A_Process",
    clean_tmp=False,
):
    """If on linux, remember to chown user by:
    sudo chown user -R /home/user/Desktop/Sen2Cor-02.09.00-Linux64/

    download and install: http://step.esa.int/main/snap-supported-plugins/sen2cor/sen2cor-v2-9/
    """
    l1_zipped = glob(l1_dir + "S2*_MSIL1C*.zip")

    unzip_files_to_folder(l1_zipped, tmp_dir)

    if gip == "default":
        use_gip = ""
    else:
        use_gip = "--GIP_L2A ./graphs/L2A_GIPP.xml "

    for l1_file in glob(tmp_dir + "S2*_MSIL1C*.SAFE"):
        call = f"sudo {sen2cor_path} --resolution {resolution} --output_dir {dst_dir} {use_gip}{os.path.abspath(l1_file)}"
        os.system(call)

    if clean_tmp:
        tmp_files = glob(tmp_dir + "*.SAFE")
        for f in tmp_files:
            try:
                rmtree(f)
            except:
                pass

    return True
