# u-net model with up-convolution or up-sampling and weighted binary-crossentropy as loss func

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, LeakyReLU, Add
from keras.optimizers import Adam
from keras.initializers import he_uniform
from keras.utils import plot_model
from keras.losses import mean_absolute_error
from keras import backend as K
from loss import *

def wnet_model(n_classes=5, im_sz=160, n_channels=3, n_filters_start=32, growth_factor=2):
    droprate=0.25

    #-------------Encoder
    #Block1
    n_filters = n_filters_start
    inputs = Input((im_sz, im_sz, n_channels), name='input')
    #inputs = BatchNormalization()(inputs)
    conv1 = Conv2D(n_filters, (3, 3),  padding='same', name = 'conv1_1')(inputs)
    actv1 = LeakyReLU(name = 'actv1_1')(conv1)
    conv1 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv1_2')(actv1)
    actv1 = LeakyReLU(name = 'actv1_2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), name = 'maxpool1')(actv1)
    #pool1 = Dropout(droprate)(pool1)

    #Block2
    n_filters *= growth_factor
    pool1 = BatchNormalization(name = 'bn1')(pool1)
    conv2 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv2_1')(pool1)
    actv2 = LeakyReLU(name = 'actv2_1')(conv2)
    conv2 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv2_2')(actv2)
    actv2 = LeakyReLU(name = 'actv2_2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name = 'maxpool2')(actv2)
    pool2 = Dropout(droprate, name = 'dropout2')(pool2)

    #Block3
    n_filters *= growth_factor
    pool2 = BatchNormalization(name = 'bn2')(pool2)
    conv3 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv3_1')(pool2)
    actv3 = LeakyReLU(name = 'actv3_1')(conv3)
    conv3 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv3_2')(actv3)
    actv3 = LeakyReLU(name = 'actv3_2')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name = 'maxpool3')(actv3)
    pool3 = Dropout(droprate, name = 'dropout3')(pool3)

    #Block4
    n_filters *= growth_factor
    pool3 = BatchNormalization(name = 'bn3')(pool3)
    conv4_0 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv4_1')(pool3)
    actv4_0 = LeakyReLU(name = 'actv4_1')(conv4_0)
    conv4_0 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv4_0_2')(actv4_0)
    actv4_0 = LeakyReLU(name = 'actv4_2')(conv4_0)
    pool4_1 = MaxPooling2D(pool_size=(2, 2), name = 'maxpool4')(actv4_0)
    pool4_1 = Dropout(droprate, name = 'dropout4')(pool4_1)

    #Block5
    n_filters *= growth_factor
    pool4_1 = BatchNormalization(name = 'bn4')(pool4_1)
    conv4_1 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv5_1')(pool4_1)
    actv4_1 = LeakyReLU(name = 'actv5_1')(conv4_1)
    conv4_1 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv5_2')(actv4_1)
    actv4_1 = LeakyReLU(name = 'actv5_2')(conv4_1)
    pool4_2 = MaxPooling2D(pool_size=(2, 2), name = 'maxpool5')(actv4_1)
    pool4_2 = Dropout(droprate, name = 'dropout5')(pool4_2)

    #Block6
    n_filters *= growth_factor
    conv5 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv6_1')(pool4_2)
    actv5 = LeakyReLU(name = 'actv6_1')(conv5)
    conv5 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv6_2')(actv5)
    actv5 = LeakyReLU(name = 'actv6_2')(conv5)

    #-------------Decoder
    #Block7
    n_filters //= growth_factor
    up6_1 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same', name = 'up7')(actv5), actv4_1], name = 'concat7')
    up6_1 = BatchNormalization(name = 'bn7')(up6_1)
    conv6_1 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv7_1')(up6_1)
    actv6_1 = LeakyReLU(name = 'actv7_1')(conv6_1)
    conv6_1 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv7_2')(actv6_1)
    actv6_1 = LeakyReLU(name = 'actv7_2')(conv6_1)
    conv6_1 = Dropout(droprate, name = 'dropout7')(actv6_1)

    #Block8
    n_filters //= growth_factor
    up6_2 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same', name = 'up8')(conv6_1), actv4_0], name = 'concat8')
    up6_2 = BatchNormalization(name = 'bn8')(up6_2)
    conv6_2 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv8_1')(up6_2)
    actv6_2 = LeakyReLU(name = 'actv8_1')(conv6_2)
    conv6_2 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv8_2')(actv6_2)
    actv6_2 = LeakyReLU(name = 'actv8_2')(conv6_2)
    conv6_2 = Dropout(droprate, name = 'dropout8')(actv6_2)

    #Block9
    n_filters //= growth_factor
    up7 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same', name = 'up9')(conv6_2), actv3], name = 'concat9')
    up7 = BatchNormalization(name = 'bn9')(up7)
    conv7 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv9_1')(up7)
    actv7 = LeakyReLU(name = 'actv9_1')(conv7)
    conv7 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv9_2')(actv7)
    actv7 = LeakyReLU(name = 'actv9_2')(conv7)
    conv7 = Dropout(droprate, name = 'dropout9')(actv7)

    #Block10
    n_filters //= growth_factor
    up8 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same', name = 'up10')(conv7), actv2], name = 'concat10')
    up8 = BatchNormalization(name = 'bn10')(up8)
    conv8 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv10_1')(up8)
    actv8 = LeakyReLU(name = 'actv10_1')(conv8)
    conv8 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv10_2')(actv8)
    actv8 = LeakyReLU(name = 'actv10_2')(conv8)
    conv8 = Dropout(droprate, name = 'dropout10')(actv8)

    #Block11
    n_filters //= growth_factor
    up9 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same', name = 'up11')(conv8), actv1], name = 'concat11')
    conv9 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv11_1')(up9)
    actv9 = LeakyReLU(name = 'actv11_1')(conv9)
    conv9 = Conv2D(n_filters, (3, 3), padding='same', name = 'conv11_2')(actv9)
    actv9 = LeakyReLU(name = 'actv11_2')(conv9)

    output1 = Conv2D(n_classes, (1, 1), activation='softmax', name = 'output1')(actv9)

    #-------------Second UNet
    #-------------Encoder
    #Block12
    conv10 = Conv2D(n_filters, (3, 3),  padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(output1)
    actv10 = LeakyReLU()(conv10)
    conv10 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv10)
    actv10 = LeakyReLU()(conv10)
    pool10 = MaxPooling2D(pool_size=(2, 2))(actv10)

    #Block13
    n_filters *= growth_factor
    pool10 = BatchNormalization()(pool10)
    #Bridge
    pool10 = concatenate([pool10, conv8])
    conv11 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(pool10)
    actv11 = LeakyReLU()(conv11)
    conv11 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv11)
    actv11 = LeakyReLU()(conv11)
    pool11 = MaxPooling2D(pool_size=(2, 2))(actv11)
    pool11 = Dropout(droprate)(pool11)


    #Block14
    n_filters *= growth_factor
    pool11 = BatchNormalization()(pool11)
    pool11 = concatenate([pool11, conv7])
    conv12 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(pool11)
    actv12 = LeakyReLU()(conv12)
    conv12 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv12)
    actv12 = LeakyReLU()(conv12)
    pool12 = MaxPooling2D(pool_size=(2, 2))(actv12)
    pool12 = Dropout(droprate)(pool12)

    #Block15
    n_filters *= growth_factor
    pool12 = BatchNormalization()(pool12)
    pool12 = concatenate([pool12, conv6_2])
    conv13_0 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(pool12)
    actv13_0 = LeakyReLU()(conv13_0)
    conv13_0 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv13_0)
    actv13_0 = LeakyReLU()(conv13_0)
    pool13_1 = MaxPooling2D(pool_size=(2, 2))(actv13_0)
    pool13_1 = Dropout(droprate)(pool13_1)

    #Block16
    n_filters *= growth_factor
    pool13_1 = BatchNormalization()(pool13_1)
    pool13_1 = concatenate([pool13_1, conv6_1])
    conv13_1 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(pool13_1)
    actv13_1 = LeakyReLU()(conv13_1)
    conv13_1 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv13_1)
    actv13_1 = LeakyReLU()(conv13_1)
    pool13_2 = MaxPooling2D(pool_size=(2, 2))(actv13_1)
    pool13_2 = Dropout(droprate)(pool13_2)

    #Block17
    n_filters *= growth_factor
    pool13_2 = concatenate([pool13_2, actv5])
    conv14 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(pool13_2)
    actv14 = LeakyReLU()(conv14)
    conv14 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv14)
    actv14 = LeakyReLU()(conv14)

    #-------------Decoder
    #Block18
    n_filters //= growth_factor
    #Skip
    up15_1 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(actv14), Add()([actv13_1, actv4_1])])

    up15_1 = BatchNormalization()(up15_1)
    conv15_1 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(up15_1)
    actv15_1 = LeakyReLU()(conv15_1)
    conv15_1 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv15_1)
    actv15_1 = LeakyReLU()(conv15_1)
    conv15_1 = Dropout(droprate)(actv15_1)

    #Block19
    n_filters //= growth_factor
    #Skip
    up15_2 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv15_1), Add()([actv13_0, actv4_0])])
    up15_2 = BatchNormalization()(up15_2)
    conv15_2 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(up15_2)
    actv15_2 = LeakyReLU()(conv15_2)
    conv15_2 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv15_2)
    actv15_2 = LeakyReLU()(conv15_2)
    conv15_2 = Dropout(droprate)(actv15_2)

    #Block20
    n_filters //= growth_factor
    #Skip
    up16 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv15_2), Add()([actv12, actv3])])
    up16 = BatchNormalization()(up16)
    conv16 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(up16)
    actv16 = LeakyReLU()(conv16)
    conv16 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv16)
    actv16 = LeakyReLU()(conv16)
    conv16 = Dropout(droprate)(actv16)

    #Block21
    n_filters //= growth_factor
    #Skip
    up17 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv16), Add()([actv11, actv2])])
    up17 = BatchNormalization()(up17)
    conv17 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(up17)
    actv17 = LeakyReLU()(conv17)
    conv17 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv17)
    actv17 = LeakyReLU()(conv17)
    conv17 = Dropout(droprate)(actv17)

    #Block22
    n_filters //= growth_factor
    #Skip
    up18 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv17), Add()([actv10, actv1])])
    conv18 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(up18)
    actv18 = LeakyReLU()(conv18)
    conv18 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv18)
    actv18 = LeakyReLU()(conv18)

    #conv19 = Conv2D(n_classes, (1, 1), activation='sigmoid')(actv18)
    conv19 = Conv2D(n_channels, (1, 1), activation='sigmoid', name = 'output2')(actv18)

    output2 = conv19

    model = Model(inputs=inputs, outputs=[output1, output2])

    def mean_squared_error(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        return K.mean(K.square(y_pred_f - y_true_f))


    def dice_coef(y_true, y_pred, smooth=1e-7):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    """This simply calculates the dice score for each individual label, and then sums them together, and includes the background."""
    def dice_coef_multilabel(y_true, y_pred):
        dice=n_classes
        for index in range(n_classes):
            dice -= dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
        return dice/n_classes

    model.compile(optimizer=Adam(lr = 10e-5), loss=[focal_tversky, mean_squared_error], loss_weights  = [0.95, 0.05])
    return model
