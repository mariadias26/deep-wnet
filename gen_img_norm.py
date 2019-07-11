import os.path
import tifffile as tiff
import numpy as np

val_ids = ['5', '7', '23', '37']
name_template = '/top_mosaic_09cm_area{}.tif'

path_img = '/home/mdias/datasets/vaihingen/Images/'

new_path_img = '/home/mdias/datasets/vaihingen/Images_norm/'


if not os.path.exists(new_path_img): os.makedirs(new_path_img)

files = [f for f in os.listdir(path_img) if os.path.isfile(os.path.join(path_img, f))]

for f in files:
    img = tiff.imread(path_img + f)
    new_img = img/255
    tiff.imsave(new_path_img+f, new_img)