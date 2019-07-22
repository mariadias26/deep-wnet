from unet_model import *
from wnet_model import *
from gen_mask_neighbor import *
from gen_patches import *
from generator import *
from clr_callback import *
import os.path
import tensorflow as tf
import sys
import decimal

from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

N_BANDS = 3
N_CLASSES = 6  # imp surface, car, building, background, low veg, tree
N_EPOCHS = 50

DATASET = 'vaihingen'  # 'vaihingen'
MODEL = 'W'
ID = '40'
#gen_mask()

if DATASET == 'potsdam':
    TRAIN_IDS = ['2_10', '2_11', '3_10', '3_11', '4_10', '4_11', '5_10', '5_11', '6_7', '6_8', '6_9', '6_10', '6_11',
                 '7_7', '7_8', '7_9', '7_10', '7_11']
    VAL_IDS = ['2_12', '3_12', '4_12', '5_12', '6_12', '7_12']
    path_img = '/home/mdias/datasets/potsdam/Images_lab_hist/top_potsdam_{}_RGB.tif'
    path_mask = '/home/mdias/datasets/potsdam/Masks/top_potsdam_{}_label.tif'
    #PATCH_SZ = 320  # should divide by 16
    PATCH_SZ = 128
    VALIDATION_STEPS = 2400
    if MODEL == 'U':
        STEPS_PER_EPOCH = 8000
        BATCH_SIZE = 32
        MAX_QUEUE = 30
    elif MODEL == 'W':
        STEPS_PER_EPOCH = 10000
        BATCH_SIZE = 12
        MAX_QUEUE = 10
elif DATASET == 'vaihingen':
    TRAIN_IDS = ['1', '3', '11', '13', '15', '17', '21', '26', '28', '30', '32', '34']#,
                 #'_rc_1', '_rc_3', '_rc_11', '_rc_13', '_rc_15', '_rc_17', '_rc_21', '_rc_26', '_rc_28', '_rc_30', '_rc_32', '_rc_34']
    VAL_IDS = ['5', '7', '23', '37']
    path_img = '/home/mdias/datasets/vaihingen/Images_lab_hist/top_mosaic_09cm_area{}.tif'
    path_full_img = '/home/mdias/datasets/vaihingen/Images_lab_hist/top_mosaic_09cm_area{}.tif'
    path_mask = '/home/mdias/datasets/vaihingen/Masks/top_mosaic_09cm_area{}.tif'

    # Val paths
    path_patch_img = '/home/mdias/datasets/vaihingen/Images_lab_hist_patch/'
    path_patch_full_img = '/home/mdias/datasets/vaihingen/Images_lab_hist_patch/'
    path_patch_mask = '/home/mdias/datasets/vaihingen/Masks_patch/'

    PATCH_SZ = 320  # should divide by 16
    BATCH_SIZE = 5
    STEPS_PER_EPOCH = 10000

    VALIDATION_STEPS = 1369
    MAX_QUEUE = 10


def get_files_weights(path, train_ids):
    files_weights = []
    count_all = 0
    for id in train_ids:
        img = tiff.imread(path.format(id))
        count = img.shape[0]*img.shape[1]
        count_all+=count
        files_weights.append(count)
    files_weights = np.array(files_weights)
    files_weights = files_weights / files_weights.sum()
    return files_weights


def get_model():
    if MODEL == 'U':
        model = unet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS)
    elif MODEL == 'W':
        model = wnet_model(N_CLASSES, PATCH_SZ, n_channels=N_BANDS)
    return model


weights_path = '/home/mdias/weights/weights_' + MODEL + '_' + DATASET + '_' + ID
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
weights_path += '/weights.hdf5'

if __name__ == '__main__':
    def train_net():
        print("start train net")
        model = get_model()
        if os.path.isfile(weights_path):
            model.load_weights(weights_path)
        early_stopping = EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=5, mode='min')
        model_checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, mode='min')
        csv_logger = CSVLogger('log_unet.csv', append=True, separator=';')
        step_size = (STEPS_PER_EPOCH // BATCH_SIZE) * 8
        clr = CyclicLR(base_lr=10e-5, max_lr=10e-4, step_size=step_size, mode='triangular2')

        files_weights = get_files_weights(path_img, TRAIN_IDS)
        train_gen = image_generator(TRAIN_IDS, path_img, path_mask, path_full_img, files_weights, batch_size = BATCH_SIZE, patch_size = PATCH_SZ)
        val_gen = val_generator(path_patch_img, path_patch_full_img, path_patch_mask, batch_size = BATCH_SIZE)

        model.fit_generator(train_gen,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            nb_epoch=N_EPOCHS,
                            validation_data=val_gen,
                            validation_steps=VALIDATION_STEPS,
                            verbose=1, shuffle=True, max_queue_size=MAX_QUEUE,
                            callbacks=[model_checkpoint, csv_logger, early_stopping, clr]
                            )

        return model


    train_net()
