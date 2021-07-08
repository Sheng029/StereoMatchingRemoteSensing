import numpy as np
from osgeo import gdal


def epe_metric(est_path, gt_path, threshold):
    """
    End-point-error calculation of an estimated disparity map.
    :param est_path: path of an estimated disparity map.
    :param gt_path: path of the corresponding ground-truth disparity map.
    :param threshold: used to determine whether a pixel is valid.
    :return: sum of absolute error, num of valid pixels, end-point-error.
    """
    est_disp = gdal.Open(est_path).ReadAsArray()
    gt_disp = gdal.Open(gt_path).ReadAsArray()

    zeros = np.zeros_like(gt_disp, 'float32')
    ones = np.ones_like(gt_disp, 'float32')
    mask = np.where(np.abs(gt_disp) > threshold, zeros, ones)

    error = np.sum(np.abs(est_disp - gt_disp) * mask)   # sum of absolute error
    nums = np.sum(mask)   # num of valid pixels
    epe = error / nums    # end-point-error

    return error, nums, epe


def d1_metric(est_path, gt_path, threshold):
    """
    D1 (<3 pixels) calculation of an estimated disparity map.
    :param est_path: path of an estimated disparity map.
    :param gt_path: path of the corresponding ground-truth disparity map.
    :param threshold: used to determine whether a pixel is valid.
    :return: num of error pixels, num of valid pixels, D1.
    """
    est_disp = gdal.Open(est_path).ReadAsArray()
    gt_disp = gdal.Open(gt_path).ReadAsArray()

    zeros = np.zeros_like(gt_disp, 'float32')
    ones = np.ones_like(gt_disp, 'float32')
    mask = np.where(np.abs(gt_disp) > threshold, zeros, ones)

    err_map = np.abs(est_disp - gt_disp) * mask
    err_mask = err_map > 3
    err_disps = np.sum(err_mask.astype('float32'))   # num of pixels whose disparity error is larger than 3
    nums = np.sum(mask)   # num of valid pixels
    d1 = err_disps / nums   # d1

    return err_disps, nums, d1
