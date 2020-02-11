import os.path
import cv2
from get_step import *
from patchify import patchify, unpatchify
import tifffile as tiff
import numpy as np
PATCH_SZ = 320
N_CLASSES= 6

val_ids = ['2_12', '3_12', '4_12', '5_12', '6_12', '7_12']

path_l = '/home/mdias/datasets/potsdam/Images_l/top_potsdam_{}_RGB.tif'
path_img = '/home/mdias/datasets/potsdam/Images_lab_hist/top_potsdam_{}_RGB.tif'
path_mask = '/home/mdias/datasets/potsdam/Masks/top_potsdam_{}_label.tif'

name_template = 'top_potsdam_{}_patch.tif'
path_patch_l = '/home/mdias/datasets/potsdam/Images_l_patch/'
path_patch_img = '/home/mdias/datasets/potsdam/Images_lab_hist_patch/'
path_patch_mask = '/home/mdias/datasets/potsdam/Masks_patch/'

paths = {path_l: path_patch_l, path_img: path_patch_img, path_mask: path_patch_mask}

for p in paths:
    if not os.path.exists(paths[p]): os.makedirs(paths[p])
    for the_file in os.listdir(paths[p]):
        file_path = os.path.join(paths[p], the_file)
        try:
            if os.path.isfile(file_path) and '.tif' in file_path and 'patch' in file_path:
                os.unlink(file_path)
        except Exception as e:
            print(e)

for id in val_ids:
    paths_origin = list(paths.keys())
    img_to_pad = tiff.imread(paths_origin[0].format(id))
    step = 71
    # step, x_padding, y_padding, x_original, y_original = find_step(img_to_pad, PATCH_SZ, id, 35, 60)
    for p in paths:
        print('\n',p.format(id))
        img = tiff.imread(p.format(id))
        x, y, dim = img.shape
        #print('Step: ', step, '\nx_padding: ', x_padding, '\ny_padding: ', y_padding, '\nx_original: ', x_original,
        #      '\ny_original: ', y_original)
        # img = img[:x_padding, :y_padding, :]
        patches = patchify(img, (PATCH_SZ, PATCH_SZ, dim), step = step)
        print(patches.shape)
        width_window, height_window, z, width_x, height_y, dim = patches.shape
        patches = np.reshape(patches, (width_window * height_window, width_x, height_y, dim))
        batch_size, x, y, dim = patches.shape
        for i in range(batch_size):
            new_patch = np.array(patches[i], copy=True)
            tiff.imsave(paths[p] + name_template.format(id + '_' + str(i)), new_patch)
