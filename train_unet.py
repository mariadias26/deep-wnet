from unet_model import *
from gen_patches import *

import os.path
import numpy as np
import tifffile as tiff
import rasterio
import glob

from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

'''
def normalize(img):
    min = img.min()
    max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x
'''


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

path_img='./potsdam/2_Ortho_RGB/*.tif'
files_img = [os.path.basename(f) for f in glob.glob(path_img)]
path_gt='./potsdam/5_Labels_all_norm/*.tif'
files_gt =  [os.path.basename(f) for f in glob.glob(path_gt)]

def get_model():
    return unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV, class_weights=CLASS_WEIGHTS)


weights_path = 'weights'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
weights_path += '/unet_weights.hdf5'

trainIds = [str(i).zfill(2) for i in range(13, 38)]  # all availiable ids: from "01" to "38"


if __name__ == '__main__':
    X_DICT_TRAIN = dict()
    Y_DICT_TRAIN = dict()
    X_DICT_VALIDATION = dict()
    Y_DICT_VALIDATION = dict()

    print('Reading images')
    dict_all = dict()
    countTrain = 0
    countVal = 0

    for train_id in TRAIN_IDS:
        for file in files_img:
            if train_id in file:
                name_img = './potsdam/2_Ortho_RGB/{}'.format(file)
                break
        #img = tiff.imread(name_img).transpose([1,2,0])
        img = rasterio.open(name_img)
        img = img.read().transpose([1,2,0])

        mask = tiff.imread('./potsdam/5_Labels_all_norm/top_potsdam_{}_label.tif'.format(train_id))
        X_DICT_TRAIN[train_id] = img
        Y_DICT_TRAIN[train_id] = mask
        countTrain+=1
        print(train_id + ' read')


    for val_id in VAL_IDS:
        for file in files_img:
            if val_id in file:
                name_img = './potsdam/2_Ortho_RGB/{}'.format(file)
                break
        #img = tiff.imread(name_img).transpose([1,2,0])
        img = rasterio.open(name_img)
        img = img.read().transpose([1,2,0])

        mask = tiff.imread('./potsdam/5_Labels_all_norm/top_potsdam_{}_label.tif'.format(train_id))
        X_DICT_VALIDATION[val_id] = img
        Y_DICT_VALIDATION[val_id] = mask
        countVal+=1
        print(val_id + ' read')
    print('Images were read')
    print('Train with ', countTrain, ' images and validation with ', countVal, ' images.')

    def train_net():
        print("start train net")
        x_train, y_train = get_patches(X_DICT_TRAIN, Y_DICT_TRAIN, n_patches=TRAIN_SZ, sz=PATCH_SZ)
        x_val, y_val = get_patches(X_DICT_VALIDATION, Y_DICT_VALIDATION, n_patches=VAL_SZ, sz=PATCH_SZ)
        model = get_model()
        if os.path.isfile(weights_path):
            model.load_weights(weights_path)
        #model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_weights_only=True, save_best_only=True)
        #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')
        #reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=0.00001)
        model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
        csv_logger = CSVLogger('log_unet.csv', append=True, separator=';')
        tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)

        #model.fit_generator(generator=training_generator,
        #                    validation_data=validation_generator,
        #                    )
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
                  verbose=2, shuffle=True,
                  callbacks=[model_checkpoint, csv_logger, tensorboard],
                  validation_data=(x_val, y_val))
        return model

    train_net()
