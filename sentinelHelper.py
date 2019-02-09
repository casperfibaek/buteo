import os, sys, rasterio
import utils
import gdal

def readS2( str ): 
    name = utils.getFilename(str)
    filetype = utils.getFiletype(str)
    path = utils.getPath(str)
    dataset = None

    # GGDAL Sentinel Parser does not read folder structure unless zipped. Below is a work around
    try:
        if filetype == 'zip':
            dataset = gdal.Open(path)
        elif filetype == 'SAFE':
            if os.path.isfile(f'{path}\\MTD_MSIL2A.xml'):
                dataset = gdal.Open(f'{path}\\MTD_MSIL2A.xml')
            else:
                dataset = gdal.Open(f'{path}\\MTD_MSIL1C.xml')
        elif filetype == 'safe' or filetype == 'xml':
            if os.path.isfile(f'{os.path.dirname(path)}\\MTD_MSIL2A.xml'):
                dataset = gdal.Open(f'{os.path.dirname(path)}\\MTD_MSIL2A.xml')
            else:
                dataset = gdal.Open(f'{os.path.dirname(path)}\\MTD_MSIL1C.xml')
        else:
            raise ValueError('Input could not be parsed' ) from None
    except:
        raise ValueError('Input could not be parsed') from None

    try:
        meta = dataset.GetMetadata_Dict()
    except:
        raise ValueError('Input could not be parsed - is the path valid?') from None

    data = {
        '10m': {},
        '20m': {}, 
        '60m': {},
        'tci': {},
        'metadata': meta
    }

    bands_10m = gdal.Open(dataset.GetSubDatasets()[0][0]).GetFileList()
    bands_20m = gdal.Open(dataset.GetSubDatasets()[1][0]).GetFileList()
    bands_60m = gdal.Open(dataset.GetSubDatasets()[2][0]).GetFileList()
    bands_tci = gdal.Open(dataset.GetSubDatasets()[3][0]).GetFileList()

    # Get all the 10m bands
    for index, value in enumerate(bands_10m):
        if (utils.getFiletype(value) == 'jp2'):
            data['10m'][utils.getFilename(value).split('_')[2]] = bands_10m[index]

    # Get all the 20m bands
    for index, value in enumerate(bands_20m):
        if (utils.getFiletype(value) == 'jp2'):
            data['20m'][utils.getFilename(value).split('_')[2]] = bands_20m[index]

    # Get all the 60m bands
    for index, value in enumerate(bands_60m):
        if (utils.getFiletype(value) == 'jp2'):
            data['60m'][utils.getFilename(value).split('_')[2]] = bands_60m[index]

    # Get all the true color bands
    for index, value in enumerate(bands_tci):
        if (utils.getFiletype(value) == 'jp2'):
            data['tci'][utils.getFilename(value).split('_')[2]] = bands_tci[index]

    # Delete all the references again to free memory.
    del dataset
    del bands_10m
    del bands_20m
    del bands_60m
    del bands_tci

    return data
