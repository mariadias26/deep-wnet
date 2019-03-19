from __future__ import print_function
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import rasterio
from train_unet_generator import weights_path, get_model, PATCH_SZ, N_CLASSES
from scipy import stats
from sklearn.metrics import classification_report, accuracy_score
import gc
from skimage.util.shape import view_as_windows
import psutil


def old_predict(x, model, patch_sz=160, n_classes=5):
    img_height = x.shape[0]
    img_width = x.shape[1]
    n_channels = x.shape[2]
    # make extended img so that it contains integer number of patches
    npatches_vertical = math.ceil(img_height / patch_sz)
    npatches_horizontal = math.ceil(img_width / patch_sz)
    extended_height = patch_sz * npatches_vertical
    extended_width = patch_sz * npatches_horizontal
    ext_x = np.zeros(shape=(extended_height, extended_width, n_channels), dtype=np.float32)
    # fill extended image with mirrors:
    ext_x[:img_height, :img_width, :] = x
    for i in range(img_height, extended_height):
        ext_x[i, :, :] = ext_x[2 * img_height - i - 1, :, :]
    for j in range(img_width, extended_width):
        ext_x[:, j, :] = ext_x[:, 2 * img_width - j - 1, :]

    # now we assemble all patches in one array
    patches_list = []
    for i in range(0, npatches_vertical):
        for j in range(0, npatches_horizontal):
            x0, x1 = i * patch_sz, (i + 1) * patch_sz
            y0, y1 = j * patch_sz, (j + 1) * patch_sz
            patches_list.append(ext_x[x0:x1, y0:y1, :])
    # model.predict() needs numpy array rather than a list
    patches_array = np.asarray(patches_list)
    # predictions:
    patches_predict = model.predict(patches_array, batch_size=4)
    prediction = np.zeros(shape=(extended_height, extended_width, n_classes), dtype=np.float32)
    for k in range(patches_predict.shape[0]):
        i = k // npatches_horizontal
        j = k % npatches_vertical
        x0, x1 = i * patch_sz, (i + 1) * patch_sz
        y0, y1 = j * patch_sz, (j + 1) * patch_sz
        prediction[x0:x1, y0:y1, :] = patches_predict[k, :, :, :]
    return prediction[:img_height, :img_width, :]


def predict(x, model, patch_sz=160, n_classes=5):
    print('\n\n', np.shape(x))
    patches = view_as_windows(x, (patch_sz, patch_sz, x.ndim), step = patch_sz//5)
    width_window, height_window, z, width_x, height_y, num_channel = patches.shape
    patches = np.reshape(patches, (width_window, height_window,  width_x, height_y, num_channel))
    print(np.shape(patches))

    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    print('memory % used:', psutil.virtual_memory()[2])
    predict_array = np.empty(shape=np.shape(patches))
    print(np.shape(predict_array))
    #for w in width_window:

    patches_predict = model.predict(patches[0], batch_size=4)
    print(np.shape(patches_predict), '\n\n\n')
    return prediction[:img_height, :img_width, :]


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

if __name__ == '__main__':
    model = get_model()
    model.load_weights(weights_path)
    test = ['2_13','2_14','3_13','3_14','4_13','4_14','4_15','5_13','5_14','5_15','6_13','6_14','6_15','7_13']
    accuracy_all = []
    for test_id in test:
        path_img = './../data-mdias/Images/top_potsdam_{}_RGB.tif'.format(test_id)
        img = tiff.imread(path_img)
        path_mask = './../data-mdias/5_Labels_all/top_potsdam_{}_label.tif'.format(test_id)
        label = tiff.imread(path_mask).transpose([2,0,1])
        gt = mask_from_picture(label)

        mask = predict(img, model, patch_sz=PATCH_SZ, n_classes=N_CLASSES).transpose([2,0,1])
        print('\n\nMask', np.shape(mask))
        prediction = picture_from_mask(mask)

        target_labels = ['imp surf', 'car', 'building', 'background', 'low veg', 'tree']
        y_true = gt.ravel()
        y_pred = np.argmax(mask, axis=0).ravel()
        report = classification_report(y_true, y_pred, target_names = target_labels)
        accuracy = accuracy_score(y_true, y_pred)
        print('\n',test_id)
        print(report)
        print('\nAccuracy', accuracy)
        accuracy_all.append(accuracy)
        gc.collect()
        gc.collect()
        gc.collect()

    print(accuracy_all)
    print('Accuracy all', sum(accuracy_all)/len(accuracy_all))
    tiff.imsave('map.tif', prediction)
