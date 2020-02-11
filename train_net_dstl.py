from unet_model import *
from wnet_model import *
from generator_dstl import *
from clr_callback import *
import os.path
import tensorflow as tf

from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

N_BANDS = 3
N_CLASSES = 7  # buildings, imp surface, trees, low veg, water, car, background
N_EPOCHS = 50

DATASET = 'dstl'  # 'vaihingen'
MODEL = 'W'
ID = '14'


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


path = '/home/mdias/datasets/dstl-satellite-imagery-feature-detection/'
TRAIN_IDS = ['6040_4_4', '6010_4_4', '6040_1_3', '6170_4_1', '6010_1_2', '6160_2_1', '6100_2_2', '6170_2_4', '6110_1_2', '6140_3_1', '6060_2_3', '6110_4_0', '6150_2_3', '6070_2_3', '6120_2_2', '6170_0_4', '6090_2_0']
VAL_IDS = ['6010_4_2', '6110_3_1', '6100_1_3']

print(TRAIN_IDS, '\n', VAL_IDS)
print('\n\n', len(TRAIN_IDS), len(VAL_IDS), '\n\n')

path_img = path + 'images_lab/{}.tif'
path_mask = path + 'mask_uni/{}.tif'

PATCH_SZ = 224  # should divide by 16
VALIDATION_STEPS = 2000

STEPS_PER_EPOCH = 8000
BATCH_SIZE = 12
MAX_QUEUE = 30


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
        #clr = CyclicLR(base_lr=10e-5, max_lr=10e-4, step_size=step_size, mode='triangular2')

        train_gen = image_generator(TRAIN_IDS, path_img, path_mask, path_img, batch_size = BATCH_SIZE, patch_size = PATCH_SZ)
        val_gen = image_generator(VAL_IDS, path_img, path_mask, path_img, batch_size=BATCH_SIZE,
                                    patch_size=PATCH_SZ)
        #val_gen = val_generator(path_patch_img, path_patch_full_img, path_patch_mask, batch_size = BATCH_SIZE)

        model.fit_generator(train_gen,
                            steps_per_epoch=STEPS_PER_EPOCH,
                            nb_epoch=N_EPOCHS,
                            validation_data=val_gen,
                            validation_steps=VALIDATION_STEPS,
                            verbose=1, shuffle=True, max_queue_size=MAX_QUEUE,
                            callbacks=[model_checkpoint, csv_logger, early_stopping]
                            )

        return model


    train_net()
