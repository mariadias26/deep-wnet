import tifffile as tiff
import numpy as np
from skimage.measure import compare_ssim

def picture_from_mask(mask):
    colors = {
        0: [255, 0, 0],
        1: [0, 255, 0],
    }

    pict = np.empty(shape=(3, mask.shape[0], mask.shape[1]))
    for cl in range(2):
      for ch in range(3):
        pict[ch,:,:] = np.where(mask == cl, colors[cl][ch], pict[ch,:,:])
    return pict

test_id = '6'
#test_id = '6_14'

#path_mask = './potsdam/test/5_Labels_all/top_potsdam_{}_label.tif'
path_mask = './vaihingen/Ground_Truth/top_mosaic_09cm_area{}.tif'

prediction = tiff.imread('./results/final_map_{}.tif'.format(test_id)).transpose([1,2,0])
mask = tiff.imread(path_mask.format(test_id))

result = (prediction == mask).all(axis=2)

pict = picture_from_mask(result.astype(int))
tiff.imsave('./results/potsdam_rg.tif', pict.transpose([0,1,2]).astype('uint8'))
