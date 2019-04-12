from unet_model import *
from gen_patches import *
from image_data_generator import ImageDataGenerator

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
GENERATED_DATA = 'statistics'

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


def get_model():
    return unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS, upconv=UPCONV, class_weights=CLASS_WEIGHTS)

def get_map(color_codes):
  color_map = np.ndarray(shape=(256*256*256), dtype='int32')
  color_map[:] = -1
  for rgb, idx in color_codes.items():
    rgb = rgb[0] * 65536 + rgb[1] * 256 + rgb[2]
    color_map[rgb] = idx
  return color_map

COLOR_MAP = get_map(color_codes)

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

    def xy_provider(image_ids, infinite=True):
        while True:
            np.random.shuffle(image_ids)
            for image_id in image_ids:
                image_name = path_img.format(image_id)
                target_name = path_mask.format(image_id)

                image = rasterio.open(image_name).read().transpose([1,2,0])
                target =  norm_image(tiff.imread(target_name), COLOR_MAP)
                print(image_id)
                print(np.shape(target))
                yield image, target
            if not infinite:
                return
    train_gen = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True,
                             rotation_range=90.,
                             width_shift_range=0.15, height_shift_range=0.15,
                             shear_range=3.14/6.0,
                             zoom_range=0.25,
                             channel_shift_range=0.1,
                             horizontal_flip=True,
                             vertical_flip=True)
    train_gen.fit(xy_provider(TRAIN_IDS, infinite=False),
            len(TRAIN_IDS))
    val_gen = ImageDataGenerator(featurewise_center=True,
                                 featurewise_std_normalization=True) # Just an infinite image/mask generator
    val_gen.mean = train_gen.mean
    val_gen.std = train_gen.std
    val_gen.principal_components = train_gen.principal_components


    def train_net():
            print("start train net")

            model = get_model()
            if os.path.isfile(weights_path):
                model.load_weights(weights_path)

            #callbacks
            early_stopping = EarlyStopping(monitor='val_loss', restore_best_weights = True, patience = 5)
            model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
            csv_logger = CSVLogger('log_unet.csv', append=True, separator=';')
            tensorboard = TensorBoard(log_dir='./tensorboard_unet/', write_graph=True, write_images=True)

            model.fit_generator(
                train_gen.flow(xy_provider(TRAIN_IDS), # Infinite generator is used
                       len(train_id_type_list),
                       batch_size=BATCH_SIZE),
               samples_per_epoch=samples_per_epoch,
               nb_epoch=N_EPOCHS,
               validation_data=val_gen.flow(xy_provider(VAL_IDS), # Infinite generator is used
                              len(val_image_ids),
                              batch_size=BATCH_SIZE),
               nb_val_samples=nb_val_samples,
               verbose=2, shuffle=True,
                callbacks=[model_checkpoint, csv_logger, tensorboard, early_stopping]
            )
            return model
    train_net()
