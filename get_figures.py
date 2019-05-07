import tifffile as tiff
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import numpy.ma as ma
import pandas as pd

test = ['2', '4', '6', '8', '10', '12', '14', '16', '20', '22', '24', '27', '29', '31', '33', '35', '38']
test_id = test[2]
test_id = '6_14'
def picture_from_mask(mask):
    colors = {
        0: [255, 255, 255],   #imp surface
        1: [255, 255, 0],     #car
        2: [0, 0, 255],       #building
        3: [255, 0, 0],       #background
        4: [0, 255, 255],     #low veg
        5: [0, 255, 0],        #tree
        6: [0, 0, 0]
    }

    mask_ind = np.argmax(mask, axis=0)
    pict = np.empty(shape=(3, mask.shape[1], mask.shape[2]))
    for cl in range(6):
      for ch in range(3):
        pict[ch,:,:] = np.where(mask_ind == cl, colors[cl][ch], pict[ch,:,:])
    return pict

def mask_from_picture(picture):
  colors = {
      (255, 255, 255): 0,   #imp surface
      (255, 255, 0): 1,     #car
      (0, 0, 255): 2,       #building
      (255, 0, 0): 3,       #background
      (0, 255, 255): 4,     #low veg
      (0, 255, 0): 5,        #tree
      (0, 0, 0): 6
  }
  picture = picture.transpose([1,2,0])
  mask = np.ndarray(shape=(256*256*256), dtype='int32')
  mask[:] = -1
  for rgb, idx in colors.items():
    rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
    mask[rgb] = idx

  picture = picture.dot(np.array([65536, 256, 1], dtype='int32'))
  return mask[picture]



path_mask_nb = './potsdam/5_Labels_all_noBoundary/top_potsdam_{}_label_noBoundary.tif'
#path_mask_nb = './vaihingen/Ground_Truth_noBoundary/top_mosaic_09cm_area{}_noBoundary.tif'
path_mask_predict = './results/mask_wnet_potsdam_{}.tif'
#path_mask_predict = './results/mask_wnet_vaihingen3_{}.tif'
mask_nb = tiff.imread(path_mask_nb.format(test_id)).transpose([2,0,1])
gt = mask_from_picture(mask_nb)
mask = tiff.imread(path_mask_predict.format(test_id))
prediction = picture_from_mask(mask).astype('uint8')
tiff.imsave('./results/final_map_{}.tif'.format(test_id), prediction)
