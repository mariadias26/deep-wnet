from __future__ import print_function
import math
import os
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import cv2
from train_net_all import weights_path, get_model, PATCH_SZ, N_CLASSES, DATASET, MODEL, ID, N_BANDS
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
from sklearn.metrics import confusion_matrix



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


def predict(x, model, step):
    dim_x, dim_y, dim = x.shape
    patches = patchify(x, (PATCH_SZ, PATCH_SZ, N_BANDS), step=step)
    width_window, height_window, z, width_x, height_y, num_channel = patches.shape
    patches = np.reshape(patches, (width_window * height_window,  width_x, height_y, num_channel))
    predictions = model.predict(patches, batch_size=40)
    patches_predict = reconstruct_patches(predictions[0], (dim_x, dim_y, N_CLASSES), step)
    return patches_predict #, image_predict


def predict_all(path_img, path_mask, ids, dataset, step = -1):
    model = get_model()
    print(weights_path)
    model.load_weights(weights_path)
    tps, tns, fns, fps = 0, 0, 0, 0
    accuracy_all = []
    path_results = '/home/mdias/datasets/results/'+MODEL+'_'+DATASET+'_'+ID
    if not os.path.exists(path_results): os.makedirs(path_results)
    for test_id in ids:
        path_i = path_img.format(test_id)
        print(path_i)
        img = tiff.imread(path_i)
        path_m = path_mask.format(test_id)
        label = tiff.imread(path_m)#.transpose([0,1])
        if dataset == 'vaihingen':
            step, x_padding, y_padding, x_original, y_original = find_step(img, PATCH_SZ, test_id)
            print('Step: ', step, x_padding, y_padding, x_original, y_original)
            img = cv2.copyMakeBorder(img, 0, x_padding-x_original, 0, y_padding-y_original, cv2.BORDER_CONSTANT)
        #mask, image_predict = predict(img, model, patch_sz=PATCH_SZ, n_classes=N_CLASSES, step = step)
        pred = predict(img, model, step)
        if dataset == 'vaihingen':
            pred = pred[:x_original, :y_original, ]
        l, x_orig, y_orig = pred.shape
        #result = np.where(pred.reshape((x_orig, y_orig)) < 0.5, 0, 1)
        result = np.where(pred < 0.5, 0, 1)

        tn, fp, fn, tp = confusion_matrix(label.ravel(), result.ravel(), labels=[0, 1]).ravel()
        fps += fp; fns += fn; tps += tp; tns += tn

        print('\n',test_id)
        print('TN: ', tn)
        print('FP: ', fp)
        print('FN: ', fn)
        print('TP: ', tp)
        acc = (tp+tn)/(tn+fp+fn+tp)
        iou_ind = tp / (tp + fn + fp)
        print('\nAccuracy', acc)
        print('\nIoU', iou_ind)
        accuracy_all.append(acc)
        print('oioi', np.shape(result), np.unique(result))
        if dataset == 'inria':
            tiff.imsave(path_results + '/mask_{}'.format(test_id), result.astype('uint8'))
        else:
            tiff.imsave(path_results + '/mask_{}.tif'.format(test_id), result.astype('uint8'))
        #tiff.imsave(path_results + '/image_{}.tif'.format(test_id), image_predict)
        sys.stdout.flush()
        sys.stdout.flush()
        sys.stdout.flush()
    accuracy = (tps + tns) / (tps + tns + fps + fns)
    iou = tps / (tps + fns + fps)
    print(accuracy, iou)
    print(step,' Accuracy all', sum(accuracy_all)/len(accuracy_all))


# Potsdam
ids_potsdam = ['2_13','2_14','3_13','3_14','4_13','4_14','4_15','5_13','5_14','5_15','6_13','6_14','6_15','7_13']
path_img_potsdam = '/home/mdias/datasets/unified/images_potsdam/top_potsdam_{}_RGB.tif'
path_mask_potsdam = '/home/mdias/datasets/unified/gt_potsdam/top_potsdam_{}_label.tif'
predict_all(path_img_potsdam, path_mask_potsdam, ids_potsdam, 'potsdam', step=8)

# INRIA
path_img_inria = '/home/mdias/datasets/unified/teste/images_inria/'
path_mask_inria = '/home/mdias/datasets/unified/teste/gt_inria/'
ids_inria = [f for f in listdir(path_img_inria) if isfile(join(path_img_inria, f))]
path_img_inria += '{}'
path_mask_inria += '{}'
predict_all(path_img_inria, path_mask_inria, ids_inria, 'inria', step = 24)

# Vaihingen
ids_vaihingen = ['2', '4', '6', '8', '10', '12', '14', '16', '20', '22', '24', '27', '29', '31', '33', '35', '38']
path_img_vaihingen = '/home/mdias/datasets/unified/images_vaihingen/top_mosaic_09cm_area{}.tif'
path_mask_vaihingen = '/home/mdias/datasets/unified/gt_vaihingen/top_mosaic_09cm_area{}.tif'
predict_all(path_img_vaihingen, path_mask_vaihingen, ids_vaihingen, 'vaihingen')
