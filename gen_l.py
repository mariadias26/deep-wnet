import tifffile as tiff
import os
from os import listdir
from os.path import isfile, join
import numpy as np

# INRIA
path_img_inria = '/home/mdias/datasets/unified/images_inria/'
path_img_inria_val = '/home/mdias/datasets/unified/validation/images_inria/'
path_img_inria_test = '/home/mdias/datasets/unified/teste/images_inria/'

ids_inria_train = [f for f in listdir(path_img_inria) if isfile(join(path_img_inria, f))]
ids_inria_val = [f for f in listdir(path_img_inria_val) if isfile(join(path_img_inria_val, f))]
ids_inria_test = [f for f in listdir(path_img_inria_test) if isfile(join(path_img_inria_test, f))]

path_img_inria += '{}'
path_img_inria_val += '{}'
path_img_inria_test += '{}'

new_path_img_inria = '/home/mdias/datasets/unified/images_inria_l/{}'
new_path_img_inria_val = '/home/mdias/datasets/unified/validation/images_inria_l/{}'
new_path_img_inria_test = '/home/mdias/datasets/unified/teste/images_inria_l/{}'

# Potsdam
ids_potsdam_train = ['2_10', '2_11', '3_10', '3_11', '4_10', '4_11', '5_10', '5_11', '6_7', '6_8', '6_9', '6_10', '6_11',
             '7_7', '7_8', '7_9', '7_10', '7_11']
ids_potsdam_val = ['2_12', '3_12', '4_12', '5_12', '6_12', '7_12']
ids_potsdam_test = ['2_13','2_14','3_13','3_14','4_13','4_14','4_15','5_13','5_14','5_15','6_13','6_14','6_15','7_13']
path_img_potsdam = '/home/mdias/datasets/unified/images_potsdam/top_potsdam_{}_RGB.tif'
new_path_img_potsdam = '/home/mdias/datasets/unified/images_potsdam_l/top_potsdam_{}_RGB.tif'

# Vaihingen
ids_vaihingen_train = ['1', '3', '11', '13', '15', '17', '21', '26', '28', '30', '32', '34']
ids_vaihingen_val = ['5', '7', '23', '37']
ids_vaihingen_test = ['2', '4', '6', '8', '10', '12', '14', '16', '20', '22', '24', '27', '29', '31', '33', '35', '38']
path_img_vaihingen = '/home/mdias/datasets/unified/images_vaihingen/top_mosaic_09cm_area{}.tif'
new_path_img_vaihingen = '/home/mdias/datasets/unified/images_vaihingen_l/top_mosaic_09cm_area{}.tif'

paths = {'ids_inria_train': [ids_inria_train, path_img_inria, new_path_img_inria],
        'ids_inria_val': [ids_inria_val, path_img_inria_val, new_path_img_inria_val],
        'ids_inria_test': [ids_inria_test, path_img_inria_test, new_path_img_inria_test],
        'ids_potsdam_train': [ids_potsdam_train, path_img_potsdam, new_path_img_potsdam],
        'ids_potsdam_val': [ids_potsdam_val, path_img_potsdam, new_path_img_potsdam],
        'ids_potsdam_test': [ids_potsdam_test, path_img_potsdam, new_path_img_potsdam],
        'ids_vaihingen_train': [ids_vaihingen_train, path_img_vaihingen, new_path_img_vaihingen],
        'ids_vaihingen_val': [ids_vaihingen_val, path_img_vaihingen, new_path_img_vaihingen],
        'ids_vaihingen_test': [ids_vaihingen_test, path_img_vaihingen, new_path_img_vaihingen]}

for p in paths:
    print(p)
    for id in paths[p][0]:
        img = tiff.imread(paths[p][1].format(id))[:,:,0]
        img = np.expand_dims(img, axis = 2)
        tiff.imsave(paths[p][2].format(id), img)
