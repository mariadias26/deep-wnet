from os import listdir
from os.path import isfile, join
import tifffile as tiff
import rasterio
import numpy as np

#dataset = input('Potsdam (p) or Vaihingen (v) dataset? ')
#if dataset == 'v':

vaihingen = './vaihingen/top/'
new_vaihingen = './vaihingen/Images/'
groundTruth = './vaihingen/GroundTruth/'
new_mask = './vaihingen/Masks/'
files_images = [f for f in listdir(vaihingen) if isfile(join(vaihingen, f))]
files_masks = [f for f in listdir(groundTruth) if isfile(join(groundTruth, f))]

for f in files_images:
    if '.tif' in f:
        img = rasterio.open(vaihingen + f).read().transpose([1,2,0])
        tiff.imsave(new_vaihingen+f, img)


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

  #image = np.rint(image/255)*255
  image = image.astype(int)
  image = image.dot(np.array([65536, 256, 1], dtype='int32'))

  new_a = color_map[image]
  image_norm = (np.arange(new_a.max()) == new_a[...,None]-1).astype(int)
  return image_norm

i = 1
for f in files_masks:
    mask = tiff.imread(groundTruth + f)
    tiff.imsave(new_mask + f, norm_image(mask))
    i+=1
