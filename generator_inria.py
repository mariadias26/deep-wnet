import numpy as np
from gen_patches_inria import *
import tifffile as tiff
import os
from os import listdir
from os.path import isfile, join

def get_input(path):
    #image = rasterio.open(path).read().transpose([1,2,0])
    image = tiff.imread(path)
    return image

def get_mask(path):
    mask = tiff.imread(path)
    return mask


def image_generator(path_img_inria, path_full_img_inria, path_mask_inria, ids_inria,
                                    path_img_potsdam, path_full_img_potsdam, path_mask_potsdam, ids_potsdam,
                                    path_img_vaihingen, path_full_img_vaihingen, path_mask_vaihingen, ids_vaihingen,
                                    batch_size = 5, patch_size = 160):
    seed = 0
    while True:
        datasets_choice = ['inria', 'potsdam', 'vaihingen']
        datasets_chance = [0.2, 0.4, 0.4]
        np.random.seed(seed)
        dataset = np.random.choice(datasets_choice, p=datasets_chance)

        if dataset == 'inria':
            path_image = path_img_inria
            path_mask = path_mask_inria
            path_full_img = path_full_img_inria
            ids_file = ids_inria

        if dataset == 'potsdam':
            path_image = path_img_potsdam
            path_mask = path_mask_potsdam
            path_full_img = path_full_img_potsdam
            ids_file = ids_potsdam

        if dataset == 'vaihingen':
            path_image = path_img_vaihingen
            path_mask = path_mask_vaihingen
            path_full_img = path_full_img_vaihingen
            ids_file = ids_vaihingen

        np.random.seed(seed)
        file_choice = np.random.choice((len(ids_file)))
        id = ids_file[file_choice]

        image = get_input(path_image.format(id))
        mask = get_mask(path_mask.format(id))
        full_img = get_input(path_full_img.format(id))
        total_patches = 0
        x = list()
        y = list()
        y2 = list()
        while total_patches < batch_size:
            img_patch, mask_patch, full_img_patch = get_rand_patch(image, mask, full_img, patch_size)
            x.append(img_patch)
            y.append(mask_patch)
            y2.append(full_img_patch)
            total_patches += 1

        batch_x = np.array( x )
        batch_y = np.array( y )
        batch_y2 = np.array( y2 )
        seed+=1
        yield ( batch_x, [batch_y , batch_y2])
        #yield ( batch_x, batch_y )


def val_generator(path_patch_img, path_patch_full_img, path_patch_mask, batch_size = 5):
    files = [f for f in listdir(path_patch_img) if isfile(join(path_patch_img, f))]

    while True:
        if len(files) < 10:
            files = [f for f in listdir(path_patch_img) if isfile(join(path_patch_img, f))]
        total_patches = 0
        x = list()
        y = list()
        y2 = list()
        while total_patches < batch_size:
            file = files.pop()
            x.append(get_input(path_patch_img+file))
            y.append(get_input(path_patch_mask+file))
            y2.append(get_input(path_patch_full_img+file))
            total_patches += 1

        batch_x = np.array(x)
        batch_y = np.array(y)
        batch_y2 = np.array(y2)
        yield (batch_x, [batch_y, batch_y2])
