from keras.losses import binary_crossentropy
import keras.backend as K
import tensorflow as tf

import tifffile as tiff
import numpy as np
from os import listdir
from os.path import isfile, join

epsilon = 1e-5
smooth = 1
'''
def confusion(y_true, y_pred):
    smooth=1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)
    prec = (tp + smooth)/(tp+fp+smooth)
    recall = (tp+smooth)/(tp+fn+smooth)
    return prec, recall

def tp(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth)/ (K.sum(y_pos) + smooth)
    return tp

def tn(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
    return tn

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)
'''

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

def get_n_instances():
    dataset = input('Potsdam (p) or Vaihingen (v) dataset? ')
    while True:
        if dataset == 'p':
            path = './datasets/potsdam/5_Labels_all/'
            break
        elif dataset == 'v':
            path = './datasets/vaihingen/Images/'
            break
        else:
            dataset = input('p or v?')
    files = [f for f in listdir(path) if isfile(join(path, f))]
    for f in files:
        filename = path + f
        img = tiff.imread(filename).transpose([2,0,1])
        mask = mask_from_picture(img)
        y = np.bincount(mask)
        ii = np.nonzero(y)[0]
        print(np.vstack((ii,y[ii])).T)
        print(mask.shape)

get_n_instances()
