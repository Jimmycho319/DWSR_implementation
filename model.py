import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.regularizers import l2

'''
"Deep wavelet prediction for image super-resolution" by
Guo, Tiantong and Mousavi, Hojjat Seyed and Vu, Tiep Huu and Monga, Vishal
DWSR model
'''

'''
can we add crops and flips to the images to add more images or should we just use more other images for training?
this model can be easily scaled up
before training, we should compute and store all the value for the dwt
'''


def get_model(input_shape=(None, None, 4)):
    # ADD WEIGHT REGULATOR
    model = tf.keras.Sequential()
    model.add(Conv2D(filters=64, kernel_size=(5,5), padding='same', strides=(1,1),
                     activation='relu', input_shape=input_shape,
                     kernel_initializer=tf.initializers.RandomNormal(stddev=tf.sqrt(2.0/9)),
                     bias_initializer='zeros', kernel_regularizer=l2(0.001)))

    for x in range(10):
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same',
                         activation='relu',
                         kernel_initializer=tf.initializers.RandomNormal(stddev=tf.sqrt(2.0/9/64)),
                         bias_initializer='zeros', kernel_regularizer=l2(0.001)))
    model.add(Conv2D(filters=4, kernel_size=(3,3), strides=(1,1), padding='same',
                     kernel_initializer=tf.initializers.RandomNormal(stddev=tf.sqrt(2.0/9/64)),
                     bias_initializer='zeros'))
    return model


def get_loss(loss_function='mse'):
    if loss_function.upper() == 'MSE':
        return tf.keras.losses.MeanSquaredError()
    elif loss_function.upper() == 'SSIM':
        def ssim(y_true, y_pred):
            return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        return ssim
    else:
        raise ValueError("{} is not a valid loss function please use `MSE` or `SSIM`".format(loss_function))


def get_optimizer(total_epochs, rate=0.01):
    #try cosine lr
    window = total_epochs//4
    boundaries = [window, window*2, window*3]
    initial_rate = rate / 4
    values = [initial_rate*4, initial_rate*3, initial_rate*2, initial_rate]
    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    return tf.keras.optimizers.Adam(learning_rate=lr)


def get_cosine_optimizer(decay_steps, initial_learning_rate=0.001):
    lr = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate=initial_learning_rate,
                                                   decay_steps=decay_steps)
    return tf.keras.optimizers.Adam(learning_rate=lr)





# todo: understand/implement weight regulator

