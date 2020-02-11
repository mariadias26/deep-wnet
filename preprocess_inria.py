import tifffile as tiff
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image

path_img = '/home/mdias/datasets/inria/AerialImageDataset/train/images/'
new_path_img = '/home/mdias/datasets/inria/AerialImageDataset/train/images_norm/'
path_gt = '/home/mdias/datasets/inria/AerialImageDataset/train/gt/'
new_path_gt = '/home/mdias/datasets/inria/AerialImageDataset/train/gt_norm/'
ids = [f for f in listdir(path_img) if isfile(join(path_img, f))]

for id in ids:
    print(id)
    #img = tiff.imread(path_img+id)
    #img = (np.array(img)/255)
    mask = Image.open(path_gt+id)
    mask = (np.array(mask)/255).astype('uint8')
    mask = np.expand_dims(mask, axis=2)
    #tiff.imsave(new_path_img+id, img)
    tiff.imsave(new_path_gt+id, mask)