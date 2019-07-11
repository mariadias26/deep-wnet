# u-net model with up-convolution or up-sampling and weighted binary-crossentropy as loss func

from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.initializers import he_uniform
from keras import losses
from lovasz_losses_tf import *
from loss import *
from se import channel_spatial_squeeze_excite


def conv2d_block(input_tensor, n_filters, init_seed=None, kernel_size=3):
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer=he_uniform(seed=init_seed),
               bias_initializer=he_uniform(seed=init_seed), padding="same")(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer=he_uniform(seed=init_seed),
               bias_initializer=he_uniform(seed=init_seed), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def wnet_model(n_classes=5, im_sz=160, n_channels=3, n_filters_start=32, growth_factor=2, droprate=0.5, init_seed=None):
    inputs = Input((im_sz, im_sz, 1))

    # -------------Encoder
    # Block1
    n_filters = n_filters_start
    actv1 = conv2d_block(inputs, n_filters, init_seed=init_seed)
    pool1 = MaxPooling2D(pool_size=(2, 2))(actv1)
    #pool1 = channel_spatial_squeeze_excite(pool1)

    # Block2
    n_filters *= growth_factor
    actv2 = conv2d_block(pool1, n_filters, init_seed = init_seed)
    pool2 = MaxPooling2D(pool_size=(2, 2))(actv2)
    #pool2 = channel_spatial_squeeze_excite(pool2)
    pool2 = Dropout(droprate)(pool2)

    # Block3
    n_filters *= growth_factor
    actv3 = conv2d_block(pool2, n_filters, init_seed=init_seed)
    pool3 = MaxPooling2D(pool_size=(2, 2))(actv3)
    #pool3 = channel_spatial_squeeze_excite(pool3)
    pool3 = Dropout(droprate)(pool3)

    # Block4
    n_filters *= growth_factor
    actv4 = conv2d_block(pool3, n_filters, init_seed=init_seed)
    pool4 = MaxPooling2D(pool_size=(2, 2))(actv4)
    #pool4 = channel_spatial_squeeze_excite(pool4)
    pool4 = Dropout(droprate)(pool4)

    # Block5
    n_filters *= growth_factor
    actv5_new = conv2d_block(pool4, n_filters, init_seed=init_seed)
    pool5 = MaxPooling2D(pool_size=(2, 2))(actv5_new)
    #pool5 = channel_spatial_squeeze_excite(pool5)
    pool5 = Dropout(droprate)(pool5)

    # Block6
    n_filters *= growth_factor
    actv6 = conv2d_block(pool5, n_filters, init_seed=init_seed)

    # -------------Decoder
    # Block7
    n_filters //= growth_factor
    up7_new = Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(actv6)
    up7_new = concatenate([up7_new, actv5_new])
    conv7_new = Dropout(droprate)(up7_new)
    actv7_new = conv2d_block(conv7_new, n_filters, init_seed=init_seed)
    #actv7_new = channel_spatial_squeeze_excite(actv7_new)

    # Block8
    n_filters //= growth_factor
    up8_new = Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(actv7_new)
    up8_new = concatenate([up8_new, actv4])
    conv8_new = Dropout(droprate)(up8_new)
    actv8_new = conv2d_block(conv8_new, n_filters, init_seed=init_seed)
    #actv8_new = channel_spatial_squeeze_excite(actv8_new)

    # Block9
    n_filters //= growth_factor
    up9_new = Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(actv8_new)
    up9_new = concatenate([up9_new, actv3])
    conv9_new = Dropout(droprate)(up9_new)
    actv9_new = conv2d_block(conv9_new, n_filters, init_seed=init_seed)
    #actv9_new = channel_spatial_squeeze_excite(actv9_new)

    # Block10
    n_filters //= growth_factor
    up10_new = Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(actv9_new)
    up10_new = concatenate([up10_new, actv2])
    conv10_new = Dropout(droprate)(up10_new)
    actv10_new = conv2d_block(conv10_new, n_filters, init_seed=init_seed)
    #actv10_new = channel_spatial_squeeze_excite(actv10_new)

    # Block11
    n_filters //= growth_factor
    up11_new = Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(actv10_new)
    up11_new = concatenate([up11_new, actv1])
    conv11_new = Dropout(droprate)(up11_new)
    actv11_new = conv2d_block(conv11_new, n_filters, init_seed=init_seed)
    #actv11_new = channel_spatial_squeeze_excite(actv11_new)

    output1 = Conv2D(n_classes, (1, 1), activation='softmax', name = 'output1')(actv11_new)

    # -------------Second UNet
    # -------------Encoder
    # Block12
    conv10 = Conv2D(n_filters, (3, 3),  padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(output1)
    actv10 = LeakyReLU()(conv10)
    conv10 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv10)
    actv10 = LeakyReLU()(conv10)
    pool10 = MaxPooling2D(pool_size=(2, 2))(actv10)

    # Block13
    n_filters *= growth_factor
    pool10 = BatchNormalization()(pool10)
    # Bridge
    pool10 = concatenate([pool10, actv10_new])
    conv11 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(pool10)
    actv11 = LeakyReLU()(conv11)
    conv11 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv11)
    actv11 = LeakyReLU()(conv11)
    pool11 = MaxPooling2D(pool_size=(2, 2))(actv11)
    pool11 = Dropout(droprate)(pool11)


    # Block14
    n_filters *= growth_factor
    pool11 = BatchNormalization()(pool11)
    pool11 = concatenate([pool11, actv9_new])
    conv12 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(pool11)
    actv12 = LeakyReLU()(conv12)
    conv12 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv12)
    actv12 = LeakyReLU()(conv12)
    pool12 = MaxPooling2D(pool_size=(2, 2))(actv12)
    pool12 = Dropout(droprate)(pool12)

    # Block15
    n_filters *= growth_factor
    pool12 = BatchNormalization()(pool12)
    pool12 = concatenate([pool12, actv8_new])
    conv13_0 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(pool12)
    actv13_0 = LeakyReLU()(conv13_0)
    conv13_0 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv13_0)
    actv13_0 = LeakyReLU()(conv13_0)
    pool13_1 = MaxPooling2D(pool_size=(2, 2))(actv13_0)
    pool13_1 = Dropout(droprate)(pool13_1)

    # Block16
    n_filters *= growth_factor
    pool13_1 = BatchNormalization()(pool13_1)
    pool13_1 = concatenate([pool13_1, actv7_new])
    conv13_1 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(pool13_1)
    actv13_1 = LeakyReLU()(conv13_1)
    conv13_1 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv13_1)
    actv13_1 = LeakyReLU()(conv13_1)
    pool13_2 = MaxPooling2D(pool_size=(2, 2))(actv13_1)
    pool13_2 = Dropout(droprate)(pool13_2)

    # Block17
    n_filters *= growth_factor
    pool13_2 = concatenate([pool13_2, actv6])
    conv14 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(pool13_2)
    actv14 = LeakyReLU()(conv14)
    conv14 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv14)
    actv14 = LeakyReLU()(conv14)

    # -------------Decoder
    # Block18
    n_filters //= growth_factor
    # Skip
    up15_1 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(actv14), Add()([actv13_1, actv5_new])])

    up15_1 = BatchNormalization()(up15_1)
    conv15_1 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(up15_1)
    actv15_1 = LeakyReLU()(conv15_1)
    conv15_1 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv15_1)
    actv15_1 = LeakyReLU()(conv15_1)
    conv15_1 = Dropout(droprate)(actv15_1)

    # Block19
    n_filters //= growth_factor
    # Skip
    up15_2 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv15_1), Add()([actv13_0, actv4])])
    up15_2 = BatchNormalization()(up15_2)
    conv15_2 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(up15_2)
    actv15_2 = LeakyReLU()(conv15_2)
    conv15_2 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv15_2)
    actv15_2 = LeakyReLU()(conv15_2)
    conv15_2 = Dropout(droprate)(actv15_2)

    # Block20
    n_filters //= growth_factor
    # Skip
    up16 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv15_2), Add()([actv12, actv3])])
    up16 = BatchNormalization()(up16)
    conv16 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(up16)
    actv16 = LeakyReLU()(conv16)
    conv16 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv16)
    actv16 = LeakyReLU()(conv16)
    conv16 = Dropout(droprate)(actv16)

    # Block21
    n_filters //= growth_factor
    # Skip
    up17 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv16), Add()([actv11, actv2])])
    up17 = BatchNormalization()(up17)
    conv17 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(up17)
    actv17 = LeakyReLU()(conv17)
    conv17 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv17)
    actv17 = LeakyReLU()(conv17)
    conv17 = Dropout(droprate)(actv17)

    # Block22
    n_filters //= growth_factor
    # Skip
    up18 = concatenate([Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding='same')(conv17), Add()([actv10, actv1])])
    conv18 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(up18)
    actv18 = LeakyReLU()(conv18)
    conv18 = Conv2D(n_filters, (3, 3), padding='same', kernel_initializer = 'he_uniform', bias_initializer = 'he_uniform')(actv18)
    actv18 = LeakyReLU()(conv18)

    conv19 = Conv2D(n_channels, (1, 1), activation='sigmoid', name = 'output2')(actv18)

    output2 = conv19

    model = Model(inputs=inputs, outputs=[output1, output2])

    def mean_squared_error(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        return K.mean(K.square(y_pred_f - y_true_f))

    def keras_lovasz_softmax(y_true, y_pred):
        zero = y_true[:, :, :, 1]*0
        two = tf.where(tf.equal(y_true[:, :, :, 1], 1), y_true[:, :, :, 1], zero)
        three = tf.where(tf.equal(y_true[:, :, :, 2], 1), y_true[:, :, :, 2]*2, zero)
        four = tf.where(tf.equal(y_true[:, :, :, 3], 1), y_true[:, :, :, 3]*3, zero)
        five = tf.where(tf.equal(y_true[:, :, :, 4], 1), y_true[:, :, :, 4]*4, zero)
        six = tf.where(tf.equal(y_true[:, :, :, 5], 1), y_true[:, :, :, 5]*5, zero)
        new_y_true = two + three + four + five + six
        return lovasz_softmax(y_pred, new_y_true)

    def custom_loss(y_true, y_pred):
        return dice_coef_multilabel(y_true, y_pred) + losses.binary_crossentropy(y_true, y_pred)

    #n_instances_per_class = [v for k, v in get_n_instances(dataset).items()]
    #model.compile(optimizer=Adam(lr=10e-5), loss=[keras_lovasz_softmax, mean_squared_error], loss_weights=[0.95,0.05])
    model.compile(optimizer=Adam(lr = 10e-5), loss=[custom_loss, mean_squared_error], loss_weights  = [0.95, 0.05], metrics=["accuracy"])
    #model.compile(optimizer=Adam(lr = 10e-5), loss=[dice_coef_multilabel, mean_squared_error], loss_weights  = [0.95, 0.05])
    #model.compile(optimizer=Adam(lr = 10e-5), loss=[categorical_class_balanced_focal_loss(n_instances_per_class, 0.99), mean_squared_error], loss_weights  = [0.95, 0.05])
    return model
