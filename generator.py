import numpy as np
from gen_patches import *
import tifffile as tiff
import rasterio

def get_input(path):
    image = rasterio.open(path).read().transpose([1,2,0])
    return image

color_codes = {
  (255, 255, 255): 1,   #imp surface
  (255, 255, 0): 2,     #car
  (0, 0, 255): 3,       #building
  (255, 0, 0): 4,       #background
  (0, 255, 255): 5,     #low veg
  (0, 255, 0): 6        #tree
}

def get_map(color_codes):
  color_map = np.ndarray(shape=(256*256*256), dtype='int32')
  color_map[:] = -1
  for rgb, idx in color_codes.items():
    rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
    color_map[rgb] = idx
  return color_map

color_map = get_map(color_codes)

def norm_image(image, color_map):
  image = np.rint(image/255)*255
  image = image.astype(int)
  image = image.dot(np.array([65536, 256, 1], dtype='int32'))

  new_a = color_map[image]
  image_norm = (np.arange(new_a.max()) == new_a[...,None]-1).astype(int)

  return image_norm

def get_mask(path):
    mask = tiff.imread(path)
    return norm_image(mask, color_map)


def image_generator(ids_file, path_image, path_mask, batch_size = 5, patch_size = 160):
    while True:
        id = np.random.choice(ids_file)
        image = get_input(path_image.format(id))
        mask = get_mask(path_mask.format(id))
        total_patches = 0
        x = list()
        y = list()
        while total_patches < batch_size:
            img_patch, mask_patch = get_rand_patch(image, mask, patch_size)
            x.append(img_patch)
            y.append(mask_patch)
            total_patches += 1

        batch_x = np.array( x )
        batch_y = np.array( y )
        yield ( batch_x, batch_y )
