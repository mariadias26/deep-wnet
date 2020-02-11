import tifffile as tiff
import numpy as np
import os

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def picture_from_mask(mask):
    colors = {
        0: [255, 0, 0],    # building
        1: [0, 255, 0],    # misc manmade structure
        2: [0, 0, 255],    # road
        3: [255, 255, 0],  # track
        4: [0, 255, 255],  # tree
        5: [255, 0, 255],  # crops
        6: [255, 125, 0],  # waterway
        7: [125, 255, 0],  # standing water
        8: [0, 255, 125],  # vehicle large
        9: [0, 125, 255],  # vehicle small
        10: [0, 0, 0]       # background
    }

    mask_ind = np.argmax(mask, axis=0)
    pict = np.empty(shape=(3, mask.shape[1], mask.shape[2]))
    for cl in range(11):
      for ch in range(3):
        pict[ch,:,:] = np.where(mask_ind == cl, colors[cl][ch], pict[ch,:,:])
    return pict

path_img = '/home/mdias/datasets/dstl-satellite-imagery-feature-detection/mask/{}.tif'
image_ids = list(listdir_nohidden('/home/mdias/datasets/dstl-satellite-imagery-feature-detection/train_geojson_v3/'))

for id in image_ids:
    print(id)
    img = tiff.imread(path_img.format(id)).transpose([2, 0, 1])
    mask = picture_from_mask(img)
    tiff.imsave('/home/mdias/datasets/dstl-satellite-imagery-feature-detection/mask_rgb/all_{}.tif'.format(id), mask)
