import sys
import os
import numpy as np
from glob import glob

sys.path.append('../lib')

from mosaic import main
import utilities


def mos(infile_dir, out_dir='D:\\PhD\\Projects\\Byggemodning\\s2\\mosaic\\'):
    infiles = glob(infile_dir + '\\*')
    # Get absolute path of input .safe files.
    infiles = sorted([os.path.abspath(i) for i in infiles])

    # Find all matching granule files
    infiles = utilities.prepInfiles(infiles, '2A')

    main(infiles, output_dir=out_dir, resolution=10)


mos('D:\\PhD\\Projects\\Byggemodning\\s2\\all\\32\\UPF')
mos('D:\\PhD\\Projects\\Byggemodning\\s2\\all\\32\\UPG')
mos('D:\\PhD\\Projects\\Byggemodning\\s2\\all\\32\\VPH')
