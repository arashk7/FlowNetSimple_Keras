from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import UpSampling2D

from keras.layers import Input, Concatenate
from keras.models import Model



def get_model():
    Input_1 = Input(shape=(100, 150, 2), name='Input_1')

    Convolution2D_1 = Convolution2D(name='Convolution2D_1', nb_col=7, nb_filter=6, border_mode='same',
                                    activation='relu', nb_row=7)(Input_1)

    MaxPooling2D_1 = MaxPooling2D(name='MaxPooling2D_1')(Convolution2D_1)

    Convolution2D_3 = Convolution2D(name='Convolution2D_3', nb_col=5, nb_filter=64, border_mode='same',
                                    activation='relu', nb_row=5)(MaxPooling2D_1)

    MaxPooling2D_3 = MaxPooling2D(name='MaxPooling2D_3')(Convolution2D_3)

    Convolution2D_4 = Convolution2D(name='Convolution2D_4', nb_col=5, nb_filter=128, border_mode='same',
                                    activation='relu', nb_row=5)(MaxPooling2D_3)

    MaxPooling2D_4 = MaxPooling2D(name='MaxPooling2D_4')(Convolution2D_4)

    Convolution2D_5 = Convolution2D(name='Convolution2D_5', nb_col=3, nb_filter=256, border_mode='same',
                                    activation='relu', nb_row=3)(MaxPooling2D_4)

    Convolution2D_6 = Convolution2D(name='Convolution2D_6', nb_col=3, nb_filter=256, border_mode='same',
                                    activation='relu', nb_row=3)(Convolution2D_5)

    MaxPooling2D_6 = MaxPooling2D(name='MaxPooling2D_6')(Convolution2D_6)

    Convolution2D_7 = Convolution2D(name='Convolution2D_7', nb_col=3, nb_filter=512, border_mode='same',
                                    activation='relu', nb_row=3)(MaxPooling2D_6)

    Convolution2D_8 = Convolution2D(name='Convolution2D_8', nb_col=3, nb_filter=512, border_mode='same',
                                    activation='relu', nb_row=3)(Convolution2D_7)

    MaxPooling2D_8 = MaxPooling2D(name='MaxPooling2D_8')(Convolution2D_8)

    Convolution2D_9 = Convolution2D(name='Convolution2D_9', nb_col=3, nb_filter=512, border_mode='same',
                                    activation='relu', nb_row=3)(MaxPooling2D_8)

    Convolution2D_10 = Convolution2D(name='Convolution2D_10', nb_col=3, nb_filter=512, border_mode='same',
                                     activation='relu', nb_row=3)(Convolution2D_9)

    MaxPooling2D_9 = MaxPooling2D(name='MaxPooling2D_9')(Convolution2D_10)

    Convolution2D_11 = Convolution2D(name='Convolution2D_11', nb_col=3, nb_filter=1024, border_mode='same',
                                     activation='relu', nb_row=3)(MaxPooling2D_9)

    Convolution2D_15 = Convolution2D(name='Convolution2D_15', nb_col=4, nb_filter=512, border_mode='same',
                                     activation='relu', nb_row=4)(Convolution2D_11)

    UpSampling2D_1 = UpSampling2D(name='UpSampling2D_1', size=(2, 1))(Convolution2D_15)

    Convolution2D_17 = Convolution2D(name='Convolution2D_17', nb_col=4, nb_filter=2, border_mode='same',
                                     activation='relu', nb_row=4)(Convolution2D_11)

    UpSampling2D_3 = UpSampling2D(name='UpSampling2D_3', size=(2, 1))(Convolution2D_17)

    merge_1 = Concatenate(axis=3)([UpSampling2D_3, Convolution2D_9, UpSampling2D_1])

    Convolution2D_18 = Convolution2D(name='Convolution2D_18', nb_col=4, nb_filter=256, border_mode='same',
                                     activation='relu', nb_row=4)(merge_1)

    UpSampling2D_4 = UpSampling2D(name='UpSampling2D_4', size=(2, 1))(Convolution2D_18)

    Convolution2D_19 = Convolution2D(name='Convolution2D_19', nb_col=4, nb_filter=2, border_mode='same',
                                     activation='relu', nb_row=4)(merge_1)

    UpSampling2D_5 = UpSampling2D(name='UpSampling2D_5', size=(2, 1))(Convolution2D_19)

    merge_2 = Concatenate()([UpSampling2D_5, UpSampling2D_4, Convolution2D_7])

    Convolution2D_20 = Convolution2D(name='Convolution2D_20', nb_col=4, nb_filter=128, border_mode='same',
                                     activation='relu', nb_row=4)(merge_2)

    UpSampling2D_6 = UpSampling2D(name='UpSampling2D_6', size=(2, 1))(Convolution2D_20)

    Convolution2D_21 = Convolution2D(name='Convolution2D_21', nb_col=4, nb_filter=2, border_mode='same',
                                     activation='relu', nb_row=4)(merge_2)

    UpSampling2D_7 = UpSampling2D(name='UpSampling2D_7', size=(2, 1))(Convolution2D_21)

    merge_3 = Concatenate()([UpSampling2D_6, UpSampling2D_7, Convolution2D_5])

    Convolution2D_23 = Convolution2D(name='Convolution2D_23', nb_col=4, nb_filter=2, border_mode='same',
                                     activation='relu', nb_row=4)(merge_3)

    UpSampling2D_9 = UpSampling2D(name='UpSampling2D_9', size=(2, 1))(Convolution2D_23)

    Convolution2D_22 = Convolution2D(name='Convolution2D_22', nb_col=4, nb_filter=64, border_mode='same',
                                     activation='relu', nb_row=4)(merge_3)

    UpSampling2D_8 = UpSampling2D(name='UpSampling2D_8', size=(2, 1))(Convolution2D_22)

    merge_4 = Concatenate()([UpSampling2D_9, UpSampling2D_8])

    Convolution2D_24 = Convolution2D(name='Convolution2D_24', nb_col=4, nb_filter=1, border_mode='same',
                                     activation='relu', nb_row=4)(merge_4)

    model = Model([Input_1], [Convolution2D_24])
    return model


from keras.optimizers import *


def get_optimizer():
    return Adadelta(lr=1e-02)


def is_custom_loss_function():
    return False


def get_loss_function():
    return 'mean_squared_error'


def get_batch_size():
    return 32


def get_num_epoch():
    return 10

