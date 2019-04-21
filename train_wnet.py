from wnet_model import *
from gen_patches import *
from generator import *
from clr_callback import *
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
N_CLASSES = 6  # imp surface, car, building, background, low veg, tree
N_EPOCHS = 50
#N_EPOCHS = 5
UPCONV = True

PATCH_SZ = 320   # should divide by 16
BATCH_SIZE = 12
STEPS_PER_EPOCH = 10000
VALIDATION_STEPS = 2400
MAX_QUEUE = 10

TRAIN_IDS = ['2_10','2_11','3_10','3_11','4_10','4_11','5_10','5_11','6_7','6_8','6_9','6_10','6_11','7_7','7_8','7_9','7_10','7_11']
VAL_IDS = ['2_12','3_12','4_12','5_12','6_12','7_12']
#TRAIN_IDS = ['2_10']
#VAL_IDS = ['2_12']

path_img = './potsdam/Images_lab/top_potsdam_{}_RGB.tif'
path_mask = '/home/mdias/deep-wnet/potsdam/Masks/top_potsdam_{}_label.tif'

#path_img = './../data-mdias/Images/top_potsdam_{}_RGB.tif'
#path_mask = './../data-mdias/Masks/top_potsdam_{}_label.tif'

def get_model():
  model = wnet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV)
  return model

weights_path = 'weights_wnet_potsdam'
if not os.path.exists(weights_path):
    os.makedirs(weights_path)

wnet_weights = weights_path + '/wnet_weights.hdf5'


if __name__ == '__main__':


    def train_net():
            print("start train net")
            model = get_model()
            if os.path.isfile(wnet_weights):
                model.load_weights(wnet_weights)
            model.load_weights( 'weights_unet2/unet_weights.hdf5', by_name = True)
            early_stopping = EarlyStopping(monitor='val_loss', restore_best_weights = True, patience = 5, mode ='min')
            model_checkpoint = ModelCheckpoint(wnet_weights, monitor='val_loss', save_best_only=True, mode = 'min')
            csv_logger = CSVLogger('log_unet.csv', append=True, separator=';')
            tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)
            step_size = (STEPS_PER_EPOCH//BATCH_SIZE)*8
            clr = CyclicLR(base_lr = 10e-5, max_lr = 10e-4, step_size = step_size, mode='triangular2')

            train_gen = image_generator(TRAIN_IDS, path_img, path_mask, batch_size = BATCH_SIZE, patch_size = PATCH_SZ)
            val_gen = image_generator(VAL_IDS, path_img, path_mask, batch_size = BATCH_SIZE, patch_size = PATCH_SZ)

            model.fit_generator(train_gen,
               steps_per_epoch=STEPS_PER_EPOCH,
               nb_epoch=N_EPOCHS,
               validation_data=val_gen,
               validation_steps=VALIDATION_STEPS,
               verbose=1, shuffle=True, max_queue_size=MAX_QUEUE,
               callbacks=[model_checkpoint, csv_logger, tensorboard, early_stopping, clr]
            )
            return model
    train_net()
