import tifffile as tiff
import numpy as np
from os import listdir
from os.path import isfile, join
from skimage.color import rgb2lab
from skimage import exposure
from PIL import Image
from tqdm import tqdm

'''
# INRIA
path_inria = '/home/mdias/datasets/inria/AerialImageDataset/train/images/'
new_path_img = '/home/mdias/datasets/unified/images_inria/'
path_gt = '/home/mdias/datasets/inria/AerialImageDataset/train/gt/'
new_path_gt = '/home/mdias/datasets/unified/gt_inria/'

ids_inria = [f for f in listdir(path_inria) if isfile(join(path_inria, f))]

for id in ids_inria:
    print(id)
    img = tiff.imread(path_inria + id)
    p2, p98 = np.percentile(img, (2, 98))
    new_img = exposure.rescale_intensity(img, in_range=(p2, p98))
    lab_img = rgb2lab(new_img)
    lab_img[:, :, 0] = lab_img[:, :, 0] / 100
    lab_img[:, :, 1] = np.interp(lab_img[:, :, 1], (-128, 128), (0, 1))
    lab_img[:, :, 2] = np.interp(lab_img[:, :, 2], (-128, 128), (0, 1))
    print(np.max(lab_img), np.min(lab_img))
    tiff.imsave(new_path_img + id, lab_img)

    mask = Image.open(path_gt+id)
    mask = (np.array(mask)/255).astype('uint8')
    mask = np.expand_dims(mask, axis=2)
    tiff.imsave(new_path_gt+id, mask)

'''
# INRIA
path_gt_vai = '/home/mdias/datasets/vaihingen/test/Masks/'
new_path_gt_vai = '/home/mdias/datasets/unified/gt_vaihingen/'

ids_val = [f for f in listdir(path_gt_vai) if isfile(join(path_gt_vai, f))]

for id in tqdm(ids_val):
    mask = tiff.imread(path_gt_vai+id)
    build = np.expand_dims(mask[:,:,2], axis=2)
    print(build.shape)
    tiff.imsave(new_path_gt_vai+id, build)


path_gt_pots = '/home/mdias/datasets/potsdam/test/Masks/'
new_path_gt_pots = '/home/mdias/datasets/unified/gt_potsdam/'

ids_pots = [f for f in listdir(path_gt_pots) if isfile(join(path_gt_pots, f))]

for id in tqdm(ids_pots):
    mask = tiff.imread(path_gt_pots+id)
    build = np.expand_dims(mask[:,:,2], axis=2)
    print(build.shape)
    tiff.imsave(new_path_gt_pots+id, build)
