from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import mark_boundaries, watershed
import os
import tifffile as tiff
dataset = input('Potsdam (p) or Vaihingen (v) dataset? ')
while True:
    if dataset == 'p':
        path_img = '/home/mdias/datasets/potsdam/Images/'
        # new_path_img = '/home/mdias/datasets/potsdam/Images_l_eq_hist/'
        break
    elif dataset == 'v':
        path_img = '/home/mdias/datasets/vaihingen/Images/'
        # new_path_img = '/home/mdias/datasets/vaihingen/Images_l_eq_hist/'
        break
    else:
        dataset = input('p or v?')

# if not os.path.exists(new_path_img): os.makedirs(new_path_img)
files = [f for f in os.listdir(path_img) if os.path.isfile(os.path.join(path_img, f))]

for f in files:
    img = tiff.imread(path_img+f)
    print(img.shape)
    gradient = sobel(rgb2gray(img))
    segments_watershed = watershed(gradient, markers=800, compactness=0.01)
    result = mark_boundaries(img, segments_watershed)
    tiff.imsave('/home/mdias/deep-wnet/1.tif', result)
    print(result.shape)
    break