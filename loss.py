from keras.losses import binary_crossentropy
import keras.backend as K
import tensorflow as tf

import tifffile as tiff
import numpy as np
from os import listdir
from os.path import isfile, join
from train_net import DATASET

epsilon = 1e-5
smooth = 1

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
    if DATASET == 'potsdam':
        path = './datasets/potsdam/5_Labels_all/'
    elif DATASET == 'vaihingen':
        path = './datasets/vaihingen/Ground_Truth/'
    files = [f for f in listdir(path) if isfile(join(path, f))]
    weights = dict()
    for f in files:
        filename = path + f
        img = tiff.imread(filename).transpose([2,0,1])
        mask = mask_from_picture(img).astype('uint8')
        dim_x, dim_y = mask.shape
        y = np.bincount(mask.ravel())
        ii = np.nonzero(y)[0]
        count = dict(zip(ii, y[ii]/(dim_x*dim_y)))
        weights = { k: weights.get(k, 0) + count.get(k, 0)  for k in set(weights) | set(count) }

    weights = {k: v/len(files) for k, v in weights.items()}
    return weights

def categorical_class_balanced_focal_loss(n_instances_per_class, beta, gamma=2.):
    """
   Parameters:
     n_instances_per_class -- numpy array containing the number of instances per class in the training dataset
     gamma -- focusing parameter for modulating factor (1-p)
     beta  -- parameter for the class balancing

   Default value:
     gamma -- 2.0 as mentioned in the paper

   References:
       Official paper: https://arxiv.org/pdf/1901.05555.pdf

   Usage:
    model.compile(
               loss=[categorical_class_balanced_focal_loss(n_instances_per_class, beta, gamma=2)],
               metrics=["accuracy"],
               optimizer=adam)
   """
    effective_num = 1.0 - np.power(beta, n_instances_per_class)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights)
    weights = K.variable(weights)

    def categorical_class_balanced_focal_loss_fixed(y_true, y_pred):
        """
       :param y_true: A tensor of the same shape as `y_pred`
       :param y_pred: A tensor resulting from a softmax
       :return: Output tensor.
       """

        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = weights * K.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    return categorical_class_balanced_focal_loss_fixed
