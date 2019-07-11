import tifffile as tiff
import numpy as np
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


train_ids = ['1', '3', '11', '13', '15', '17', '21', '26', '28', '30', '32', '34', '5', '7', '23', '37']
train_ids = ['1', '3']
path_mask = '/home/mdias/datasets/vaihingen/Masks'
new_path_mask = '/home/mdias/datasets/vaihingen/Masks_neighbor'
name_template = '/top_mosaic_09cm_area{}.tif'

mask_id = '1'
mask_origin = tiff.imread(path_mask + name_template.format(mask_id)).transpose([2,0,1])
mask_neighbor = tiff.imread(new_path_mask + name_template.format(mask_id)).transpose([2,0,1])
mask_origin = picture_from_mask(mask_origin)
mask_neighbor = picture_from_mask(mask_neighbor)
print(mask_origin.shape, mask_neighbor.shape)
val = (mask_neighbor == mask_origin)


