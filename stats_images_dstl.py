import tifffile as tiff
import numpy as np
import os

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

labels_name = {0: 'building', 1: 'imp surface', 2: 'trees', 3: 'low vegetation', 4: 'water',
                5: 'car', 6: 'background', 7: 'oi'}

path_img = '/home/mdias/datasets/dstl-satellite-imagery-feature-detection/mask_uni/{}.tif'
image_ids = list(listdir_nohidden('/home/mdias/datasets/dstl-satellite-imagery-feature-detection/train_geojson_v3/'))
labels = list(range(0, 7))
labels_count_all = dict()


for el in labels:
    if el!=6:
        labels_count_all[el] = dict()
    else:
        labels_count_all[el] = 0

for id in image_ids:
    print('\n', id)
    labels_count = dict()
    mask = tiff.imread(path_img.format(id))
    for l in labels_count_all:
        if l == 6:
            labels_count_all[l] += mask.shape[0]*mask.shape[1]
        else:
            unique, counts = np.unique(mask[:, :, l], return_counts=True)
            unique_counts = [[np.round(unique[u], decimals = 2), counts[u]] for u in range(len(unique))]
            labels_count[labels_name[l]] = [[np.round(unique[u], decimals = 2), np.round(counts[u]/(mask.shape[0]*mask.shape[1]), decimals = 6)] for u in range(len(unique))]

            for x in unique_counts:
                if x[0] not in labels_count_all[l]:
                    labels_count_all[l][x[0]] = x[1]
                else:
                    labels_count_all[l][x[0]] += x[1]

    for k, v in labels_count.items():
        print(k, v)

for l in range(6):
    labels_count_all[l] = {k: v / labels_count_all[6] for k, v in labels_count_all[l].items()}

print('\n\nAll counts\n')
for k, v in labels_count_all.items():
    print(labels_name[k], v)
