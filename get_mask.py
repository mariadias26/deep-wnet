import tifffile as tiff
import numpy as np
from os import listdir
from os.path import isfile, join


def mask_from_picture(picture):
    colors = {
    (255, 255, 255): 0,   #imp surface
    (255, 255, 0): 1,     #car
    (0, 0, 255): 2,       #building
    (255, 0, 0): 3,       #background
    (0, 255, 255): 4,     #low veg
    (0, 255, 0): 5        #tree
    }
    picture = picture.transpose([1,2,0])
    mask = np.ndarray(shape=(256*256*256), dtype='int32')
    mask[:] = -1
    for rgb, idx in colors.items():
        rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
        mask[rgb] = idx
    picture = picture.dot(np.array([65536, 256, 1], dtype='int32'))
    return mask[picture]

dataset = input('Potsdam (p) or Vaihingen (v) dataset? ')
while True:
    if dataset == 'p':
        path_img = './datasets/potsdam/Ground_Truth/'
        new_path_img = './datasets/potsdam/y_true/'
        break
    elif dataset == 'v':
        path_img = './datasets/vaihingen/Ground_Truth/'
        new_path_img = './datasets/vaihingen/y_true/'
        break
    else:
        dataset = input('p or v?')

files = [f for f in listdir(path_img) if isfile(join(path_img, f))]
for f in files:
    if '.tif' in f:
        img = tiff.imread(path_img+f).transpose([2,0,1])
        new_img = mask_from_picture(img).astype('uint8')
        tiff.imsave(new_path_img+f, new_img)
