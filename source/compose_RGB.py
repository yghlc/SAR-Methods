#!/usr/bin/env python
# Filename: compose_RGB.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 09 March, 2022
"""

import os,sys

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, code_dir)

from Image import Image
from Image import SaveData
from Cluster import Cluster

import numpy as np

sys.path.insert(0, os.path.expanduser('~/codes/PycharmProjects/DeeplabforRS'))
import raster_io


# both args are Image objects
def construct_rgb_composite(VV_img, VH_img):

    VH = VH_img
    VV = VV_img

   # convert linear to dB
    VH.convert_to_dB()
    VV.convert_to_dB()

    # make the ratio band
    ratio_band = np.divide(VH.band, VV.band)
    ratio_band = np.where(VV.band == 0., 0., ratio_band)  # handle divide by zero
    # ratio_band = np.where(VH_band == 0., 0., ratio_band)
    ratio = Image(None)
    ratio.band = ratio_band
    print('\n before quantile_clip')
    print('VH,max, min, mean', np.max(VH.band), np.min(VH.band), np.mean(VH.band))
    print('VV,max, min, mean', np.max(VV.band), np.min(VV.band), np.mean(VV.band))
    print('ratio,max, min, mean', np.max(ratio.band), np.min(ratio.band), np.mean(ratio.band))

    # quantile clipping
    print('\n after quantile_clip')
    VH.quantile_clip(upper_quantile=0.99)
    VV.quantile_clip(upper_quantile=0.99)
    ratio.quantile_clip(upper_quantile=0.99)
    print('VH,max, min, mean', np.max(VH.band), np.min(VH.band), np.mean(VH.band))
    print('VV,max, min, mean', np.max(VV.band), np.min(VV.band), np.mean(VV.band))
    print('ratio,max, min, mean', np.max(ratio.band), np.min(ratio.band), np.mean(ratio.band))

    # map to interval [0,1]
    VH.map_to_interval(0, 1)
    VV.map_to_interval(0, 1)
    ratio.map_to_interval(0, 1)

    print('\n after map_to_interval')
    print('VH,max, min, mean', np.max(VH.band), np.min(VH.band), np.mean(VH.band))
    print('VV,max, min, mean', np.max(VV.band), np.min(VV.band), np.mean(VV.band))
    print('ratio,max, min, mean', np.max(ratio.band), np.min(ratio.band), np.mean(ratio.band))

    # return a numpy array because Image objects are only tested for single channel images
    rgb = np.zeros((ratio.band.shape[0], ratio.band.shape[1], 3))
    rgb[:, :, 0] = VH.band
    rgb[:, :, 1] = VV.band
    rgb[:, :, 2] = ratio.band

    return rgb

def main():
    # merge the two frames first:
    # gdal_merge.py -n 0 -a_nodata 0  -o 20170829_VH.tif */*VH*.tif
    # gdal_merge.py -n 0 -a_nodata 0  -o 20170829_VV.tif */*VV*.tif

    # require a lot of memory, run on tesia
    # data_dir=os.path.expanduser('~/Data/flooding_area/Houston/compose_3bands_VH_VV_VHVV/RGB_VH_VV_data')
    data_dir=os.path.expanduser('~/Bhaltos2/lingcaoHuang/flooding_area/Houston/compose_3bands_VH_VV_VHVV/RGB_VH_VV_data')
    VH_path = os.path.join(data_dir,'20170829_VH.tif')
    VV_path = os.path.join(data_dir,'20170829_VV.tif')
    # VH_path = os.path.join(data_dir,'20170829_VH_0.00089831528412.tif')
    # VV_path = os.path.join(data_dir,'20170829_VV_0.00089831528412.tif')

    VH_img = Image(VH_path)
    VV_img = Image(VV_path)
    VH_img.read()
    VV_img.read()

    # geographic_bounds = [-118.7, -117.7, 33.5, 34.2]
    # cluster = Cluster(geographic_bounds)

    rgb = construct_rgb_composite(VV_img, VH_img)
    print('rgb.shape',rgb.shape)
    rgb = rgb.transpose((2,0,1))        # reshape to band, height, width for GDAL
    print('after transpose rgb.shape', rgb.shape)
    rgb = rgb.astype(np.float32)
    # save to disk
    RGB_path = os.path.join(data_dir,'20170829_RGB_composite_v3.tif')
    # RGB_img = Image(RGB_path)
    raster_io.save_numpy_array_to_rasterfile(rgb,RGB_path,VH_path)


if __name__ == '__main__':
    main()