import numpy as np
import tifffile as tiff
import os

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


def picture_from_mask(mask):
    colors = {
        0: [255, 0, 0],   #buildings
        1: [0, 255, 0],   #misc manmade structures
        2: [0, 0, 255],   #road
        3: [255, 255, 0], #track
        4: [0, 255, 255], #trees
        5: [255, 0, 255], #crops
        6: [255, 125, 0], #waterway
        7: [125, 255, 0], #standing water
        8: [0, 255, 125], #vehicle large
        9: [0, 125, 255]  #vehicle small
    }

    mask_ind = np.argmax(mask, axis=0)
    pict = np.empty(shape=(3, mask.shape[1], mask.shape[2]))
    for cl in range(len(colors)):
      for ch in range(3):
        pict[ch,:,:] = np.where(mask_ind == cl, colors[cl][ch], pict[ch,:,:])
    return pict


path = '/home/mdias/datasets/dstl/'
image_ids = list(listdir_nohidden(path + 'train_geojson_v3/'))
mask = tiff.imread(path + '/mask/{}.tif'.format(image_ids[0]))
pict = picture_from_mask(mask)
print(pict.shape)
tiff.imsave('/home/mdias/deep-wnet/pict_dstl.tif', pict)
