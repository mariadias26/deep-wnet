from unet_model import *
from gen_patches import *
from generator import *
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
N_EPOCHS = 50
UPCONV = True
PATCH_SZ = 320   # should divide by 16
BATCH_SIZE = 20
STEPS_PER_EPOCH = 8000
VALIDATION_STEPS = 2000

TRAIN_IDS = ['2_10','2_11','3_10','3_11','4_10','4_11','5_10','5_11','6_7','6_8','6_9','6_10','6_11','7_7','7_8','7_9','7_10','7_11']
VAL_IDS = ['2_12','3_12','4_12','5_12','6_12','7_12']
#TRAIN_IDS = ['2_10']
#VAL_IDS = ['2_12']

path_img = './potsdam/2_Ortho_RGB/top_potsdam_{}_RGB.tif'
path_mask = './potsdam/5_Labels_all/top_potsdam_{}_label.tif'

#path_img = './../data-mdias/2_Ortho_RGB/2_Ortho_RGB/top_potsdam_{}_RGB.tif
#path_mask = './../data-mdias/5_Labels_all_norm/top_potsdam_{}_label.tif'

def get_model():
    return unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV, class_weights=CLASS_WEIGHTS)

weights_path = 'weights'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
weights_path += '/unet_weights.hdf5'


if __name__ == '__main__':


    def train_net():
            print("start train net")
            model = get_model()
            if os.path.isfile(weights_path):
                model.load_weights(weights_path)
            early_stopping = EarlyStopping(monitor='val_loss', restore_best_weights = True, patience = 5, mode ='min')
            model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, mode = 'min')
            csv_logger = CSVLogger('log_unet.csv', append=True, separator=';')
            tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)

            train_gen = image_generator(TRAIN_IDS, path_img, path_mask, batch_size = BATCH_SIZE, patch_size = PATCH_SZ)
            val_gen = image_generator(VAL_IDS, path_img, path_mask, batch_size = BATCH_SIZE, patch_size = PATCH_SZ)

            model.fit_generator(train_gen,
               steps_per_epoch=STEPS_PER_EPOCH,
               nb_epoch=N_EPOCHS,
               validation_data=val_gen,
               validation_steps=VALIDATION_STEPS,
               verbose=1, shuffle=True,
               callbacks=[model_checkpoint, csv_logger, tensorboard, early_stopping]
            )
            return model
    train_net()
