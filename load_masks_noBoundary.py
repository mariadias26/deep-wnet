import numpy as np
from os import listdir
from os.path import isfile, join
import rasterio
import tifffile as tiff

potsdam = './potsdam/5_Labels_all_noBoundary/'
#new_potsdam = './potsdam/Masks/'
files = [f for f in listdir(potsdam) if isfile(join(potsdam, f))]

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

diff_colors = []
for f in files:
    print(f)
    break
    diff_colors_f = []
    name = potsdam+f
    #new_name = new_potsdam+f
    mask = tiff.imread(name)
    width, height, cl = np.shape(mask)
    for x in range(width):
        for y in range(height):
            color = (mask[x,y,0],mask[x,y,1],mask[x,y,2])
            diff_colors_f.append(color)
            diff_colors.append(color)
    print(f, list(set(diff_colors_f)))
print(list(set(diff_colors)))
    #tiff.imsave(new_name, norm_image(mask))
