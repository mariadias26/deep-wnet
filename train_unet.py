from unet_model import *
from gen_patches import *

import os.path
import numpy as np
import tifffile as tiff
import rasterio
import glob
import gc

from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau



N_BANDS = 3
N_CLASSES = 6  # buildings, roads, trees, crops and water
CLASS_WEIGHTS = [0.15,0.15,0.15,0.15,0.2,0.2]
N_EPOCHS = 150
UPCONV = True
PATCH_SZ = 160   # should divide by 16
BATCH_SIZE = 100
TRAIN_SZ = 4000  # train size
VAL_SZ = 1000    # validation size

TRAIN_IDS = ['2_10','2_11','3_10','3_11','4_10','4_11','5_10','5_11','6_7','6_8','6_9','6_10','6_11','7_7','7_8','7_9','7_10','7_11']
VAL_IDS = ['2_12','3_12','4_12','5_12','6_12','7_12']
#TRAIN_IDS = ['2_10']
#VAL_IDS = ['2_12']

color_codes = {
  (255, 255, 255): 1,   #imp surface
  (255, 255, 0): 2,     #car
  (0, 0, 255): 3,       #building
  (255, 0, 0): 4,       #background
  (0, 255, 255): 5,     #low veg
  (0, 255, 0): 6        #tree
}

path_img = './potsdam/2_Ortho_RGB/top_potsdam_{}_RGB.tif'
path_mask = './potsdam/5_Labels_all/top_potsdam_{}_label.tif'

#path_img = './../data-mdias/2_Ortho_RGB/2_Ortho_RGB/top_potsdam_{}_RGB.tif
#path_mask = './../data-mdias/5_Labels_all_norm/top_potsdam_{}_label.tif'

def get_model():
    return unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV, class_weights=CLASS_WEIGHTS)

def get_map(color_codes):
  color_map = np.ndarray(shape=(256*256*256), dtype='int32')
  color_map[:] = -1
  for rgb, idx in color_codes.items():
    rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
    color_map[rgb] = idx
  return color_map

def norm_image(image, color_map):
  image = np.rint(image/255)*255
  image = image.astype(int)
  image = image.dot(np.array([65536, 256, 1], dtype='int32'))

  new_a = color_map[image]
  image_norm = (np.arange(new_a.max()) == new_a[...,None]-1).astype(int)

  return image_norm


weights_path = 'weights'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
weights_path += '/unet_weights.hdf5'


if __name__ == '__main__':
    X_DICT_TRAIN = dict()
    Y_DICT_TRAIN = dict()
    X_DICT_VALIDATION = dict()
    Y_DICT_VALIDATION = dict()

    print('Reading images')
    dict_all = dict()
    color_map = get_map(color_codes)

    for train_id in TRAIN_IDS:
        name_img = path_img.format(train_id)
        img = rasterio.open(name_img)
        img = img.read().transpose([1,2,0])

        mask = tiff.imread(path_mask.format(train_id))
        X_DICT_TRAIN[train_id] = img
        Y_DICT_TRAIN[train_id] = norm_image(mask, color_map)
        gc.collect()
        print(train_id, ' read')


    for val_id in VAL_IDS:

        name_img = path_img.format(val_id)
        img = rasterio.open(name_img)
        img = img.read().transpose([1,2,0])

        mask = tiff.imread(path_mask.format(val_id))
        X_DICT_VALIDATION[val_id] = img
        Y_DICT_VALIDATION[val_id] = norm_image(mask, color_map)
        gc.collect()
        print(val_id, ' read')
    print('Images were read')

    def train_net():
            print("start train net")
            x_train, y_train = get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, n_patches=TRAIN_SZ, sz=PATCH_SZ)
            x_val, y_val = get_patches(X_DICT_VALIDATION, Y_DICT_VALIDATION, n_patches=VAL_SZ, sz=PATCH_SZ)
            model = get_model()
            if os.path.isfile(weights_path):
                model.load_weights(weights_path)
            #model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_weights_only=True, save_best_only=True)
            early_stopping = EarlyStopping(monitor='val_loss', restore_best_weights = True, patience = 5)
            #reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0.00001)
            model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
            csv_logger = CSVLogger('log_unet.csv', append=True, separator=';')
            tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)
            model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                      verbose=2, shuffle=True,
                      callbacks=[model_checkpoint, csv_logger, tensorboard, early_stopping],
                      validation_data=(x_val, y_val))
            return model
    train_net()
