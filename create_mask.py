
import numpy as np
import tifffile as tiff
img = tiff.imread("/home/mdias/datasets/results/W_vaihingen_40/mask_8.tif")
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


new_img = picture_from_mask(img)
tiff.imsave("/home/mdias/datasets/results/W_vaihingen_40/mask_8_pred.tif",new_img)
