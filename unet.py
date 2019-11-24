import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


def conv_block(x, n_filters):
    x = Conv2D(n_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def create_model(n_classes=5, size=(1024, 1024)):
    inputs = Input(shape=(size[0], size[1], 6))
    x = inputs

    # Begin contractive layers

    x = conv_block(x, 64)
    x = conv_block(x, 64)
    skip_1 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = conv_block(x, 128)
    x = conv_block(x, 128)
    skip_2 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = conv_block(x, 256)
    x = conv_block(x, 256)
    skip_3 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = conv_block(x, 512)
    x = conv_block(x, 512)
    skip_4 = x
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Base of the "U"

    x = conv_block(x, 1024)
    x = conv_block(x, 1024)

    # Begin expansive layers TODO: Switch from UpSampling to ConvTranspose

    x = UpSampling2D((2, 2))(x)
    x = Concatenate(axis=-1)([skip_4, x])
    x = conv_block(x, 512)
    x = conv_block(x, 512)

    x = UpSampling2D((2, 2))(x)
    x = Concatenate(axis=-1)([skip_3, x])
    x = conv_block(x, 256)
    x = conv_block(x, 256)

    x = UpSampling2D((2, 2))(x)
    x = Concatenate(axis=-1)([skip_2, x])
    x = conv_block(x, 128)
    x = conv_block(x, 128)

    x = UpSampling2D((2, 2))(x)
    x = Concatenate(axis=-1)([skip_1, x])
    x = conv_block(x, 64)
    x = conv_block(x, 64)

    x = Conv2D(n_classes, (1, 1), padding="same")(x)

    return Model(inputs=inputs, outputs=x)
