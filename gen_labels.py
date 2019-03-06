from unet_model import *
from gen_patches import *

import os.path
import numpy as np
import tifffile as tiff
import rasterio
import glob
import gc

TRAIN_IDS = ['2_10','2_11','3_10','3_11','4_10','4_11','5_10','5_11','6_7','6_8','6_9','6_10','6_11','7_7','7_8','7_9','7_10','7_11']
VAL_IDS = ['2_12','3_12','4_12','5_12','6_12','7_12']
def get_map(color_codes):
  color_map = np.ndarray(shape=(256*256*256), dtype='int32')
  color_map[:] = -1
  for rgb, idx in color_codes.items():
    rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
    color_map[rgb] = idx
  return color_map

def norm_image(image):
  color_codes = {
      (255, 255, 255): 1,   #imp surface
      (255, 255, 0): 2,     #car
      (0, 0, 255): 3,       #building
      (255, 0, 0): 4,       #background
      (0, 255, 255): 5,     #low veg
      (0, 255, 0): 6        #tree
  }

  color_map = get_map(color_codes)

  image = np.rint(image/255)*255
  image = image.astype(int)
  image = image.dot(np.array([65536, 256, 1], dtype='int32'))

  new_a = color_map[image]
  del image


  image_norm = (np.arange(new_a.max()) == new_a[...,None]-1).astype(int)
  del new_a

  return image_norm


for train_id in TRAIN_IDS:
    mask = tiff.imread('./potsdam/5_Labels_all/top_potsdam_{}_label.tif'.format(train_id))
    tiff.imsave('./potsdam/5_Labels_all_norm/top_potsdam_{}_label.tif'.format(train_id), norm_image(mask))
    print(train_id, ' read')


for val_id in VAL_IDS:
    mask = tiff.imread('./potsdam/5_Labels_all/top_potsdam_{}_label.tif'.format(val_id))
    tiff.imsave('./potsdam/5_Labels_all_norm/top_potsdam_{}_label.tif'.format(val_id), norm_image(mask))
    print(val_id, ' read')
