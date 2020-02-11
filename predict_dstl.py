from __future__ import print_function
import math
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#import tensorflow as tf
import sys
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import cv2
from train_net_dstl import weights_path, get_model, path_img, PATCH_SZ, N_CLASSES, DATASET, MODEL, ID
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


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

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
    patches = np.reshape(patches, (width_window * height_window,  width_x, height_y, 3))
    predict = model.predict(patches, batch_size=50)
    patches_predict = predict[0]
    patches_image = predict[1]
    prediction = reconstruct_patches(patches_predict, (dim_x, dim_y, n_classes), step)
    image_prediction = reconstruct_patches(patches_image, (dim_x, dim_y, 3), step)
    return prediction, image_prediction


def picture_from_mask(mask):
    colors = {
        6: [0, 0, 0],     # background
        0: [255, 0, 0],    # building
        1: [0, 255, 0],    # imp surface
        2: [0, 0, 255],    # trees
        3: [255, 255, 0],  # low vegetation
        4: [0, 255, 255],  # water
        5: [255, 0, 255]  # car
    }

    mask_ind = np.argmax(mask, axis=0)
    pict = np.empty(shape=(3, mask.shape[1], mask.shape[2]))
    for cl in range(7):
      for ch in range(3):
        pict[ch,:,:] = np.where(mask_ind == cl, colors[cl][ch], pict[ch,:,:])
    return pict

def mask_from_picture(picture):
    colors = {
        (255, 0, 0): 0,  # building
        (0, 255, 0): 1,  # imp surface
        (0, 0, 255): 2,  # trees
        (255, 255, 0): 3,  # low vegetation
        (0, 255, 255): 4,  # water
        (255, 0, 255): 5,  # car
        (0, 0, 0): 6  # background
    }
    picture = picture.transpose([1,2,0])
    mask = np.ndarray(shape=(256*256*256), dtype='int32')
    mask[:] = -1
    for rgb, idx in colors.items():
        rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
        mask[rgb] = idx
    picture = picture.dot(np.array([65536, 256, 1], dtype='int32'))
    return mask[picture]

def gt_from_mask(mask):
    mask = picture_from_mask(mask.transpose([2, 0, 1])).astype(int)
    gt = mask_from_picture(mask)
    return gt


def predict_all(step, path_img):
    model = get_model()
    print(weights_path)
    model.load_weights(weights_path)

    path = '/home/mdias/datasets/dstl-satellite-imagery-feature-detection/'
    test = ['6040_1_0', '6040_2_2', '6140_1_2', '6120_2_0', '6100_2_3']
    path_m = path + 'mask_uni/{}.tif'
    path_i = path_img
    accuracy_all = []
    path_results = '/home/mdias/datasets/results/'+MODEL+'_'+DATASET+'_'+ID
    if not os.path.exists(path_results): os.makedirs(path_results)
    for test_id in test:
        path_img = path_i.format(test_id)
        img = tiff.imread(path_img)
        path_mask = path_m.format(test_id)
        gt = tiff.imread(path_mask)
        tiff.imsave(path_results+'/mask_true_{}.tif', np.argmax(picture_from_mask(gt.transpose([2,0,1])), axis = 0))
        gt = gt_from_mask(gt)
        step, x_padding, y_padding, x_original, y_original = find_step(img, PATCH_SZ, test_id)
        print('Step: ', step, x_padding, y_padding, x_original, y_original)
        img = cv2.copyMakeBorder(img, 0, x_padding-x_original, 0, y_padding-y_original, cv2.BORDER_CONSTANT)
        mask, image_predict = predict(img, model, patch_sz=PATCH_SZ, n_classes=N_CLASSES, step = step)
        mask = mask.transpose([2,0,1])
        mask = mask.transpose([1,2,0])
        mask = mask[:x_original, :y_original, ]
        mask = mask.transpose([2,0,1])

        prediction = picture_from_mask(mask)
        target_labels = ['buildings', 'imp surface', 'trees', 'low vegetation', 'water', 'car', 'background']
        labels = list(range(len(target_labels)))
        y_true = gt.ravel()
        y_pred = np.argmax(mask, axis=0).ravel()
        print('y_true\n', y_true.shape)
        print('y_pred', y_pred.shape)
        print('target labels', len(target_labels))
        print('labels', labels)
        report = classification_report(y_true, y_pred, target_names = target_labels, labels = labels)
        accuracy = accuracy_score(y_true, y_pred)
        print('\n',test_id)
        print(report)
        print('\nAccuracy', accuracy)
        accuracy_all.append(accuracy)
        tiff.imsave(path_results + '/mask_{}.tif'.format(test_id), picture_from_mask(mask))
        tiff.imsave(path_results + '/image_{}.tif'.format(test_id), image_predict)
        gc.collect()
        gc.collect()
        gc.collect()
        sys.stdout.flush()
    print(accuracy_all)
    print(step,' Accuracy all', sum(accuracy_all)/len(accuracy_all))




step = 40
print(step)
predict_all(step, path_img)
