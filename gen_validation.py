import os.path
import cv2
from get_step import *
from train_net import PATCH_SZ, N_CLASSES
from patchify import patchify, unpatchify
import tifffile as tiff


val_ids = ['5', '7', '23', '37']
name_template = '/top_mosaic_09cm_area{}.tif'

path_img = '/home/mdias/datasets/vaihingen/Images_l'
path_full_img = '/home/mdias/datasets/vaihingen/Images_lab_hist'
path_mask = '/home/mdias/datasets/vaihingen/Masks'
path_mask_neighbor = '/home/mdias/datasets/vaihingen/Masks_neighbor'
path_img_norm = '/home/mdias/datasets/vaihingen/Images_norm'

path_patch_img = '/home/mdias/datasets/vaihingen/Images_l_patch'
path_patch_full_img = '/home/mdias/datasets/vaihingen/Images_lab_hist_patch'
path_patch_mask = '/home/mdias/datasets/vaihingen/Masks_patch'
path_patch_mask_neighbor = '/home/mdias/datasets/vaihingen/Masks_neighbor_patch'
path_patch_img_norm = '/home/mdias/datasets/vaihingen/Images_norm_patch'

paths = {path_img: path_patch_img, path_full_img: path_patch_full_img, path_mask: path_patch_mask,
         path_mask_neighbor: path_patch_mask_neighbor, path_img_norm: path_patch_img_norm}

for p in paths:
    if not os.path.exists(paths[p]): os.makedirs(paths[p])

for id in val_ids:
    paths_origin = list(paths.keys())
    img_to_pad = tiff.imread(paths_origin[0] + name_template.format(id))
    step, x_padding, y_padding, x_original, y_original = find_step(img_to_pad, PATCH_SZ, id, 35, 60)
    for p in paths:
        print('\n',p + name_template.format(id))
        img = tiff.imread(p + name_template.format(id))
        x, y, dim = img.shape
        print('Step: ', step, '\nx_padding: ', x_padding, '\ny_padding: ', y_padding, '\nx_original: ', x_original,
              '\ny_original: ', y_original)
        img = img[:x_padding, :y_padding, :]
        patches = patchify(img, (PATCH_SZ, PATCH_SZ, dim), step = step)
        print(patches.shape)
        width_window, height_window, z, width_x, height_y, dim = patches.shape
        patches = np.reshape(patches, (width_window * height_window, width_x, height_y, dim))
        batch_size, x, y, dim = patches.shape
        for i in range(batch_size):
            tiff.imsave(paths[p] + name_template.format(id+ '_' + str(i)), patches[i])
