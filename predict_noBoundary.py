import tifffile as tiff
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import numpy.ma as ma
import pandas as pd

test = ['2_13','2_14','3_13','3_14','4_13','4_14','4_15','5_13','5_14','5_15','6_13','6_14','6_15','7_13']
#test = ['2', '4', '6', '8', '10', '12', '14', '16', '20', '22', '24', '27', '29', '31', '33', '35', '38']

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
#path_mask_predict = './results/mask_unet_vaihingen_{}.tif'
accuracy_all = []
all_reports = []
for test_id in test:
    mask_nb = tiff.imread(path_mask_nb.format(test_id)).transpose([2,0,1])
    gt = mask_from_picture(mask_nb)
    mask = tiff.imread(path_mask_predict.format(test_id))
    prediction = picture_from_mask(mask)

    target_labels = ['imp surf', 'car', 'building','low veg', 'tree']
    labels = list(range(len(target_labels)))
    y_true = gt.ravel()
    y_pred = np.argmax(mask, axis=0).ravel()

    new_y_true = y_true[np.where(y_true!=6)]
    new_y_pred = y_pred[np.where(y_true != 6)]

    new_new_y_true = new_y_true[np.where(new_y_true!=3)]
    new_new_y_pred = new_y_pred[np.where(new_y_true != 3)]

    new_new_y_true[new_new_y_true == 4] = 3
    new_new_y_true[new_new_y_true == 5] = 4

    new_new_y_pred[new_new_y_pred == 4] = 3
    new_new_y_pred[new_new_y_pred == 5] = 4


    report = classification_report(new_new_y_true, new_new_y_pred, target_names = target_labels, labels = labels, output_dict=True)
    print(classification_report(new_new_y_true, new_new_y_pred, target_names = target_labels, labels = labels))
    all_reports.append(report)
    accuracy = accuracy_score(new_new_y_true, new_new_y_pred)
    accuracy_all.append(accuracy)
    print(test_id)
    print(accuracy,'\n')

print('PRECISION')
i_s = 0
car =0
building =0
lv = 0
tree = 0
for i in all_reports:
    for j in i.keys():
        for k in i[j]:
            if k == 'precision':
                if j == 'imp surf':
                    i_s+=i[j][k]
                elif j=='car':
                    car+=i[j][k]
                elif j=='building':
                    building+=i[j][k]
                elif j=='low veg':
                    lv+=i[j][k]
                elif j=='tree':
                    tree+=i[j][k]

print('i_s', i_s/len(test))
print('car', car/len(test))
print('building', building/len(test))
print('low veg', lv/len(test))
print('tree', tree/len(test))
print('macro', (i_s/len(test)+car/len(test)+building/len(test)+ lv/len(test)+tree/len(test))/5)

print('RECALL')
i_s = 0
car =0
building =0
lv = 0
tree = 0
for i in all_reports:
    for j in i.keys():
        for k in i[j]:
            if k == 'recall':
                if j == 'imp surf':
                    i_s+=i[j][k]
                elif j=='car':
                    car+=i[j][k]
                elif j=='building':
                    building+=i[j][k]
                elif j=='low veg':
                    lv+=i[j][k]
                elif j=='tree':
                    tree+=i[j][k]

print('i_s', i_s/len(test))
print('car', car/len(test))
print('building', building/len(test))
print('low veg', lv/len(test))
print('tree', tree/len(test))
print('macro', (i_s/len(test)+car/len(test)+building/len(test)+ lv/len(test)+tree/len(test))/5)

print('F1 - SCORE')
i_s = 0
car =0
building =0
lv = 0
tree = 0
for i in all_reports:
    for j in i.keys():
        for k in i[j]:
            if k == 'f1-score':
                if j == 'imp surf':
                    i_s+=i[j][k]
                elif j=='car':
                    car+=i[j][k]
                elif j=='building':
                    building+=i[j][k]
                elif j=='low veg':
                    lv+=i[j][k]
                elif j=='tree':
                    tree+=i[j][k]

print('i_s', i_s/len(test))
print('car', car/len(test))
print('building', building/len(test))
print('low veg', lv/len(test))
print('tree', tree/len(test))
print('macro', (i_s/len(test)+car/len(test)+building/len(test)+ lv/len(test)+tree/len(test))/5)

print(' Accuracy all', sum(accuracy_all)/len(accuracy_all))
