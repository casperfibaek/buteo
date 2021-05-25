import sys

sys.path.append("../../")

from buteo.earth_observation.s2_utils import (
    get_tile_files_from_safe,
    get_metadata,
    get_tile_files_from_safe_zip,
    unzip_files_to_folder,
)
from buteo.utils import execute_cli_function


from glob import glob
from shutil import rmtree
import os

def l2_process(
    l1_dir,
    tmp_dir,
    dst_dir,
    gip="default",
    resolution=10,
    sen2cor_path="/home/cfi/Desktop/Sen2Cor-02.09.00-Linux64/bin/L2A_Process",
):
    """ If on linux, remember to chown user by:
        sudo chown user -R /home/user/Desktop/Sen2Cor-02.09.00-Linux64/

        download and install: http://step.esa.int/main/snap-supported-plugins/sen2cor/sen2cor-v2-9/
    """
    l1_zipped = glob(l1_dir + "S2*_MSIL1C*.zip")

    unzip_files_to_folder(l1_zipped, tmp_dir)

    if gip == "default":
        use_gip = ""
    else:
        use_gip = "--GIP_L2A ./L2A_GIPP.xml "

    for l1_file in glob(tmp_dir + "S2*_MSIL1C*.SAFE"):
        call = f"sudo {sen2cor_path} --resolution {resolution} --output_dir {dst_dir} {use_gip}{os.path.abspath(l1_file)}"
        os.system(call)
        # execute_cli_function(call, "L2_Process")
        # import pdb; pdb.set_trace()

    tmp_files = glob(tmp_dir + "*.SAFE")
    for f in tmp_files:
        try:
            rmtree(f)
        except:
            pass

    return True



if __name__ == "__main__":
    folder = "/home/cfi/Desktop/sentinel2/raw_l1/"

    l1_dir = folder
    tmp = folder + "tmp/"
    dst = folder + "l2/"


    l2_process(l1_dir, tmp, dst)
