import os
from glob import glob
import numpy as np
import numpy.ma as ma
import zipfile
from resample import resample
from raster_to_array import rasterToArray
from array_to_raster import arrayToRaster


def addToHolder(arr, res, holdDict):
    for band in arr:
        base = os.path.basename(band)
        filename = base.split('.')[0]
        bandname = filename.split('_')[2]
        holdDict[res][bandname] = os.path.abspath(os.path.normpath(band))


def sentinelMetadataFromFilename(filename):
    arr = filename.split('_')
    holder = {}

    s2sats = ['S2A', 'S2B']
    s1sats = ['S1A', 'S1B']

    try:
        sat = arr[0]
        if sat in s1sats:
            holder['satellite'] = sat
            holder['mode'] = arr[1]
            holder['producttype'] = arr[2]
            holder['processinglevel'] = arr[3]
            holder['sensingstart'] = arr[4]
            holder['sensingend'] = arr[5]
            holder['orbit'] = arr[6]
            holder['missionid'] = arr[7]
            holder['uid'] = arr[8]

        if sat in s2sats:
            holder['satellite'] = sat
            holder['processinglevel'] = arr[1]
            holder['sensingdate'] = arr[2]
            holder['baseline'] = arr[3]
            holder['orbit'] = arr[4]
            holder['tile'] = arr[5]
            holder['basename'] = f"{arr[5]}_{arr[2]}"
    except:
        raise ValueError('Unable to parse input string')

    return holder


def readS2(url, delete_on_unzip=False, create_nodata=False, resample_to_10m=False):
    base = os.path.basename(url)
    filetype = base.split('.')[-1]
    filename = base.split('.')[0]
    norm = os.path.normpath(url)
    currDir = os.path.dirname(norm)
    absPath = os.path.abspath(norm)

    # First parse the urlstring for metadata
    metadata = sentinelMetadataFromFilename(filename)
    if metadata['satellite'] != 'S2A' and metadata['satellite'] != 'S2B':
        if metadata['processinglevel'] != 'MSIL2A':
            raise ValueError('Level 1 data not yet supported.')

    if filetype == 'SAFE':

        imgBase = glob(f'{url}/GRANULE/*/IMG_DATA/')[0]
    elif filetype == 'zip':
        # Test if the file is already unzipped
        unzippedVersion = os.path.join(currDir, f'{filename}.SAFE')
        exists = os.path.exists(unzippedVersion)

        if exists is True:
            imgBase = glob(f'{unzippedVersion}/GRANULE/*/IMG_DATA/')[0]
        else:
            zipped = zipfile.ZipFile(absPath)
            zippedFilename = zipped.namelist()[0]
            zipped.extractall(currDir)
            zipped.close()
            if delete_on_unzip is True:
                os.remove(url)
            unzippedpath = os.path.join(currDir, zippedFilename)
            imgBase = glob(f'{unzippedpath}/GRANULE/*/IMG_DATA/')[0]
    else:
        raise ValueError('The input url could not be parsed. Is it point to the top level folder?')

    holder = {
        '10m': {},
        '20m': {},
        '60m': {},
        'meta': metadata,
        'folders': {
            'base': os.path.abspath(imgBase),
            '10m': os.path.abspath(glob(f'{imgBase}\\R10m\\')[0]),
            '20m': os.path.abspath(glob(f'{imgBase}\\R20m\\')[0]),
            '60m': os.path.abspath(glob(f'{imgBase}\\R60m\\')[0]),
        },
    }

    addToHolder(glob(f'{imgBase}\\R10m\\*'), '10m', holder)
    addToHolder(glob(f'{imgBase}\\R20m\\*'), '20m', holder)
    addToHolder(glob(f'{imgBase}\\R60m\\*'), '60m', holder)

    return holder
