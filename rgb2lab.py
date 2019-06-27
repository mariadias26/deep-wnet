import os
from os import listdir
from os.path import isfile, join
import tifffile as tiff
import numpy as np
from tqdm import tqdm
from skimage.color import rgb2lab
from skimage import exposure


dataset = input('Potsdam (p) or Vaihingen (v) dataset? ')
while True:
    if dataset == 'p':
        path_img = '/home/mdias/datasets/potsdam/Images/'
        new_path_img = '/home/mdias/datasets/potsdam/Images_l/'
        break
    elif dataset == 'v':
        path_img = '/home/mdias/datasets/vaihingen/Images/'
        new_path_img = '/home/mdias/datasets/vaihingen/Images_l/'
        break
    else:
        dataset = input('p or v?')

if not os.path.exists(new_path_img): os.makedirs(new_path_img)
files = [f for f in listdir(path_img) if isfile(join(path_img, f))]
for f in tqdm(files):
    if '.tif' in f:
        img = tiff.imread(path_img+f)
        p2, p98 = np.percentile(img, (2, 98))
        new_img = exposure.rescale_intensity(img, in_range=(p2, p98))
        lab_img = rgb2lab(new_img)
        lab_img[:,:,0] = lab_img[:,:,0]/100
        lab_img[:,:,1] = np.interp(lab_img[:,:,1], (-128, 128), (0, 1))
        lab_img[:,:,2] = np.interp(lab_img[:,:,2], (-128, 128), (0, 1))
        lab_img = lab_img[:,:,0]
        lab_img = np.expand_dims(lab_img, axis=-1)
        tiff.imsave(new_path_img+f, lab_img)
