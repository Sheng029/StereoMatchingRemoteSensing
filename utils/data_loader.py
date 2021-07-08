import numpy as np
from osgeo import gdal


def read_tif(file_name):
    """
    Read a tif file.
    :param file_name: path of tif file.
    :return: (height, width, 1) if bands = 1, otherwise (height, width, bands)
    """
    file = gdal.Open(file_name)
    if not file:
        raise Exception('Can not open.')

    data = file.ReadAsArray()
    if file.RasterCount == 1:
        # return disparity map
        return np.expand_dims(data, -1)
    else:
        # return normalized (to [-1, 1]) rgb image
        return np.transpose(data, (1, 2, 0)).astype('float32') / 127.5 - 1.0
