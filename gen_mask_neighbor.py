import tifffile as tiff
import os
import numpy as np

def gen_mask():
    train_ids = ['1', '3', '11', '13', '15', '17', '21', '26', '28', '30', '32', '34', '5', '7', '23', '37']

    path_mask = '/home/mdias/datasets/vaihingen/Masks'
    new_path_mask = '/home/mdias/datasets/vaihingen/Masks_neighbor'
    name_template = '/top_mosaic_09cm_area{}.tif'

    neighbors = [(-1, 1), (0, 1), (1, 1), (-1, 0), (1, 0), (-1, -1), (0, -1), (1, -1)]

    if not os.path.exists(new_path_mask):
        os.makedirs(new_path_mask)

    for mask_id in train_ids:
        print(mask_id)
        mask = tiff.imread(path_mask+name_template.format(mask_id)).astype(float)
        x, y, dim = mask.shape
        for i in range(x):
            for j in range(y):
                if len(np.argwhere(mask[i, j] == 1)) == 1:
                    label = np.argwhere(mask[i, j] == 1)[0][0]
                else:
                    break
                for n in neighbors:
                    n_x = i + n[0]
                    n_y = j + n[1]
                    if 0 <= n_x < x and 0 <= n_y < y:
                        if mask[n_x, n_y, label] == 0:
                            mask[i, j, label] = 0.75
                            mask[i, j, :label] = 0.25 / 5
                            mask[i, j, label+1:] = 0.25 / 5
        tiff.imsave(new_path_mask+name_template.format(mask_id), mask)


gen_mask()