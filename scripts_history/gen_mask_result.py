import tifffile as tiff
import numpy as np

def mask_from_picture(picture):
  colors = {
      (255, 255, 255): 0,   #imp surface
      (255, 255, 0): 1,     #car
      (0, 0, 255): 2,       #building
      (255, 0, 0): 3,       #background
      (0, 255, 255): 4,     #low veg
      (0, 255, 0): 5        #tree
  }
  picture = picture.transpose([1,2,0])
  mask = np.ndarray(shape=(256*256*256), dtype='int32')
  mask[:] = -1
  for rgb, idx in colors.items():
    rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
    mask[rgb] = idx

  picture = picture.dot(np.array([65536, 256, 1], dtype='int32'))
  return mask[picture]

def picture_from_mask(mask):
    colors = {
        0: [255, 255, 255],   #imp surface
        1: [255, 255, 0],     #car
        2: [0, 0, 255],       #building
        3: [255, 0, 0],       #background
        4: [0, 255, 255],     #low veg
        5: [0, 255, 0]        #tree
    }

    mask_ind = np.argmax(mask, axis=0)
    pict = np.empty(shape=(3, mask.shape[1], mask.shape[2]))
    for cl in range(6):
      for ch in range(3):
        pict[ch,:,:] = np.where(mask_ind == cl, colors[cl][ch], pict[ch,:,:])
    return pict

id = '2_14'

path_mask = './potsdam/5_Labels_all/top_potsdam_{}_label.tif'.format(id)
label = tiff.imread(path_mask).transpose([2,0,1])
gt = mask_from_picture(label)

prediction = tiff.imread('./results/map_{}.tif'.format(id))
mask = mask_from_picture(prediction)
print(np.shape(mask), np.shape(gt))
