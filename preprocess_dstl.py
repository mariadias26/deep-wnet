import os
from os import listdir
from os.path import isfile, join
import tifffile as tiff
import numpy as np
#from osgeo import gdal
from tqdm import tqdm
from skimage.color import rgb2lab
from skimage import exposure
import cv2


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


path_img = '/home/mdias/datasets/dstl-satellite-imagery-feature-detection/images_rgb_new/{}.tif'
new_path_img = '/home/mdias/datasets/dstl-satellite-imagery-feature-detection/images_lab/{}.tif'

image_ids = list(listdir_nohidden('/home/mdias/datasets/dstl/train_geojson_v3/'))
print(image_ids)
if not os.path.exists(new_path_img): os.makedirs(new_path_img)

for the_file in os.listdir(new_path_img):
    file_path = os.path.join(new_path_img, the_file)
    try:
        if os.path.isfile(file_path) and '.tif' in file_path:
            os.unlink(file_path)
    except Exception as e:
        print(e)
        
for f in image_ids:
    print(f)
    img = tiff.imread(path_img.format(f))
    p2, p98 = np.percentile(img, (2, 98))
    new_img = exposure.rescale_intensity(img, in_range=(p2, p98))
    lab_img = rgb2lab(new_img)
    lab_img[:, :, 0] = lab_img[:, :, 0] / 100
    lab_img[:, :, 1] = np.interp(lab_img[:, :, 1], (-128, 128), (0, 1))
    lab_img[:, :, 2] = np.interp(lab_img[:, :, 2], (-128, 128), (0, 1))
    print(np.max(lab_img), np.min(lab_img))
    tiff.imsave(new_path_img.format(f), lab_img)
