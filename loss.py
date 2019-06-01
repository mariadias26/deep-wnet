from keras.losses import binary_crossentropy, mean_absolute_error

import keras.backend as K
from keras.layers import Conv2D
import tensorflow as tf

import tifffile as tiff
import numpy as np
from os import listdir
from os.path import isfile, join

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

def get_n_instances(dataset):
    if dataset == 'potsdam':
        path = './datasets/potsdam/5_Labels_all/'
    elif dataset == 'vaihingen':
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

def dice_coef(y_true, y_pred, smooth=1e-9):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1.0 - ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

"""This simply calculates the dice score for each individual label, and then sums them together, and includes the background."""
def dice_coef_multilabel(y_true, y_pred, n_classes=6):
    dice = 0.0
    for index in range(n_classes): dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    if dice == 0: return dice
    return dice / n_classes

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

        dice = dice_coef_multilabel(y_true, y_pred)
        # Sum the losses in mini_batch
        return (K.sum(loss, axis=1) + dice)/2

    return categorical_class_balanced_focal_loss_fixed

def gaussian(x, mu, sigma):
    return np.exp(-(float(x) - float(mu)) ** 2 / (2 * sigma ** 2))

def make_kernel(sigma):
    # kernel radius = 2*sigma, but minimum 3x3 matrix
    kernel_size = max(3, int(2 * 2 * sigma + 1))
    mean = np.floor(0.5 * kernel_size)
    kernel_1d = np.array([gaussian(x, mean, sigma) for x in range(3*kernel_size)])
    # make 2D kernel
    np_kernel = np.outer(kernel_1d, kernel_1d).astype(dtype=K.floatx())
    # normalize kernel by sum of elements
    kernel = np_kernel / np.sum(np_kernel)
    kernel = np.reshape(kernel, (kernel_size, kernel_size, 3, 3))    #height, width, in_channels, out_channel
    return kernel

def keras_SSIM_cs(y_true, y_pred):
    axis=None
    gaussian = make_kernel(1.5)
    x = tf.nn.conv2d(y_true, gaussian, strides=[1, 1, 1, 1], padding='SAME')
    y = tf.nn.conv2d(y_pred, gaussian, strides=[1, 1, 1, 1], padding='SAME')

    u_x=K.mean(x, axis=axis)
    u_y=K.mean(y, axis=axis)

    var_x=K.var(x, axis=axis)
    var_y=K.var(y, axis=axis)

    cov_xy = K.mean(x*y, axis=axis) - u_x*u_y
    #cov_xy=cov_keras(x, y, axis)

    K1=0.01
    K2=0.03
    L=1  # depth of image (255 in case the image has a differnt scale)

    C1=(K1*L)**2
    C2=(K2*L)**2
    C3=C2/2

    l = ((2*u_x*u_y)+C1) / (K.pow(u_x,2) + K.pow(u_x,2) + C1)
    c = ((2*K.sqrt(var_x)*K.sqrt(var_y))+C2) / (var_x + var_y + C2)
    s = (cov_xy+C3) / (K.sqrt(var_x)*K.sqrt(var_y) + C3)

    return [c,s,l]

def keras_MS_SSIM(y_true, y_pred):
    iterations = 5
    x=y_true
    y=y_pred
    weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    c=[]
    s=[]
    for i in range(iterations):
        cs=keras_SSIM_cs(x, y)
        c.append(cs[0])
        s.append(cs[1])
        l=cs[2]
    c = tf.stack(c)
    s = tf.stack(s)
    cs = c*s

    l=(l+1)/2
    cs=(cs+1)/2

    cs=cs**weight
    cs = tf.reduce_prod(cs)
    l=l**weight[-1]

    ms_ssim = l*cs
    ms_ssim = tf.where(tf.is_nan(ms_ssim), K.zeros_like(ms_ssim), ms_ssim)

    return K.mean(ms_ssim)

def mix(y_true, y_pred, alpha = 0.84):
    return alpha * keras_MS_SSIM(y_true, y_pred) + (1 - alpha) * mean_absolute_error(y_true, y_pred)
