from __future__ import print_function
import math
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import cv2
from get_step import find_step
from scipy import stats
from sklearn.metrics import classification_report, accuracy_score
import gc
from skimage.util.shape import view_as_windows
import time
from itertools import product
import numbers
from sklearn.feature_extraction import image
from patchify import patchify, unpatchify
from numpy.lib.stride_tricks import as_strided
from wnet_model import *

def reconstruct_patches(patches, image_size, step):
    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    img = np.zeros(image_size)
    patch_count = np.zeros(image_size)
    # compute the dimensions of the patches array
    n_h = int((i_h - p_h) / step + 1)
    n_w = int((i_w - p_w) / step + 1)
    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
        img[i * step:i * step + p_h, j * step:j * step + p_w] += p
        patch_count[i * step:i * step + p_h, j * step:j * step + p_w] += 1
    print('MAX time seen', np.amax(patch_count))
    return img/patch_count

def predict(x, model, patch_sz=160, n_classes=5, step = 142):
    dim_x, dim_y, color_ch = x.shape
    patches = patchify(x, (patch_sz, patch_sz, color_ch), step = step)
    print('\n\n', patches.shape)
    width_window, height_window, color_ch_b, width_x, height_y, color_ch_q = patches.shape
    patches = np.reshape(patches, (width_window * height_window,  width_x, height_y, 1))
    predict = model.predict(patches, batch_size=1)
    patches_predict = predict[0]
    patches_image = predict[1]
    prediction = reconstruct_patches(patches_predict, (dim_x, dim_y, n_classes), step)
    image_prediction = reconstruct_patches(patches_image, (dim_x, dim_y, 3), step)
    return prediction, image_prediction

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
  print('\n\n',picture.shape,'\n\n')
  mask = np.ndarray(shape=(256*256*256), dtype='int32')
  mask[:] = -1
  for rgb, idx in colors.items():
    rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
    mask[rgb] = idx

  picture = picture.dot(np.array([65536, 256, 1], dtype='int32'))
  return mask[picture]

weights_path = '/home/mdias/weights/weights_W_potsdam_41/weights.hdf5'
model = wnet_model(6, 320, n_channels=3)
print(weights_path)
#print(model.summary())
model.load_weights(weights_path)
path_mask = '/home/mdias/datasets/potsdam/test/5_Labels_all/top_potsdam_4_13_label.tif'
path_img = '/home/mdias/datasets/potsdam/Images_l/top_potsdam_4_13_RGB.tif'
img = tiff.imread(path_img)
label = tiff.imread(path_mask).transpose([2, 0, 1])
print('\nshape img', img.shape)
print('\nshape label', label.shape, '\n')
gt = mask_from_picture(label)
print('\n gt', gt.shape)
result_predict = predict(img, model, patch_sz=320, n_classes=6, step =40) #40
mask, image_predict = result_predict[0], result_predict[1]
mask = mask.transpose([2,0,1])
prediction = picture_from_mask(mask)
target_labels = ['imp surf', 'car', 'building', 'background', 'low veg', 'tree']
labels = list(range(len(target_labels)))
y_true = gt.ravel()
y_pred = np.argmax(mask, axis=0).ravel()
report = classification_report(y_true, y_pred, target_names = target_labels, labels = labels)
accuracy = accuracy_score(y_true, y_pred)
print(report)
print('\nAccuracy', accuracy)

path_results = '/home/mdias/datasets/results/W_potsdam_39'
tiff.imsave(path_results + '/mask_{}.tif'.format('4_13'), mask)
tiff.imsave(path_results + '/image_{}.tif'.format('4_13'), image_predict)
