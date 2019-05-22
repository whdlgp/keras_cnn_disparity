from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# about keras
import keras
from keras.layers import Dense, Dropout
from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.layers import ZeroPadding2D, BatchNormalization, Activation
from keras.layers import UpSampling2D 
from keras.layers import concatenate
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, LambdaCallback, CSVLogger
from keras.models import load_model, Model
from keras.layers.pooling import MaxPooling2D
from keras.utils import plot_model

# about image handle library
import numpy as np
import argparse
import os
from os import path
import time
import matplotlib.image as img
import matplotlib.pyplot as plt
from scipy import misc
from skimage import io

# tensorflow video setting
# this part need if hardware gpu memory source is not enough
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7 #try various numbers here
set_session(tf.Session(config=config))

# setup for learning
EPOCH = 50
BATCH_SIZE = 4
LR = 0.0006

# setup data path for train and test
data_set_path = os.path.join('..', 'data_set')
checkpoint_path = 'check_point'

test_set_path = os.path.join(data_set_path,'test')
train_set_path = os.path.join(data_set_path, 'train')

test_disp_path = os.path.join(test_set_path,'disparity')
test_left_path = os.path.join(test_set_path,'left')
test_right_path = os.path.join(test_set_path,'right')

train_disp_path = os.path.join(train_set_path,'disparity')
train_left_path = os.path.join(train_set_path,'left')
train_right_path = os.path.join(train_set_path,'right')


# Load image from folder
def load_images_from_folder(folder):
    all_images = []
    set_list = os.listdir(folder)
    set_list.sort(key=lambda x: os.path.splitext(x)[0])
    for set_path in set_list:
        img = io.imread(os.path.join(folder,set_path))
        img = img/255
        all_images.append(img)
    return np.array(all_images)


def disparity_cnn_model(input_shape):
    shape=(None, input_shape[1], input_shape[2], input_shape[3])
    left = Input(batch_shape=shape)
    right = Input(batch_shape=shape)

    left_1 = Conv2D(filters=32, kernel_size=3, padding='same')(left)
    left_1_pool = MaxPooling2D(2)(left_1)
    left_1_activate = Activation('relu')(left_1_pool)

    left_2 = Conv2D(filters=62, kernel_size=3, padding='same')(left_1_activate)
    left_2_pool = MaxPooling2D(2)(left_2)
    left_2_activate = Activation('relu')(left_2_pool)

    left_3 = Conv2D(filters=92, kernel_size=3, padding='same')(left_2_activate)
    left_3_activate = Activation('relu')(left_3)

    right_1 = Conv2D(filters=32, kernel_size=3, padding='same')(right)
    right_1_pool = MaxPooling2D(2)(right_1)
    right_1_activate = Activation('relu')(right_1_pool)

    right_2 = Conv2D(filters=62, kernel_size=3, padding='same')(right_1_activate)
    right_2_pool = MaxPooling2D(2)(right_2)
    right_2_activate = Activation('relu')(right_2_pool)

    right_3 = Conv2D(filters=92, kernel_size=3, padding='same')(right_2_activate)
    right_3_activate = Activation('relu')(right_3)

    merge = concatenate([left_3_activate, right_3_activate])

    merge_1 = Conv2D(filters=62, kernel_size=3, padding='same')(merge)
    merge_1_up = UpSampling2D(2)(merge_1)
    merge_1_activate = Activation('relu')(merge_1_up)

    merge_2 = Conv2D(filters=22, kernel_size=3, padding='same')(merge_1_activate)
    merge_2_up = UpSampling2D(2)(merge_2)
    merge_2_activate = Activation('relu')(merge_2_up)

    merge_3 = Conv2D(filters=1, kernel_size=2, padding='same')(merge_2_activate)
    merge_3_activate = Activation('relu')(merge_3)

    model = Model([left, right], merge_3_activate)
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    return model

# Load train and test set from folder path
print('Load train_left')
train_left = load_images_from_folder(train_left_path)
print('Load train_right')
train_right = load_images_from_folder(train_right_path)
print('Load train_disp')
train_disp = load_images_from_folder(train_disp_path)
train_disp = np.expand_dims(train_disp, axis=3)
print('Load test_left')
test_left = load_images_from_folder(test_left_path)
print('Load test_right')
test_right = load_images_from_folder(test_right_path)
print('Load test_disp')
test_disp = load_images_from_folder(test_disp_path)
test_disp = np.expand_dims(test_disp, axis=3)

print(train_left.shape)
print(train_right.shape)
print(train_disp.shape)
print(test_left.shape)
print(test_right.shape)
print(test_disp.shape)

# build model
model = disparity_cnn_model(test_left.shape)
model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(lr=LR))

# Add Learning option and learning
checkpoint = ModelCheckpoint(filepath = os.path.join(checkpoint_path, 'checkpoint.h5'),
                             save_weights_only = True,
                             verbose = 1,
                             save_best_only = False)
logger = CSVLogger(filename='log.csv')
history = model.fit([train_left, train_right],
                    train_disp,
                    epochs = EPOCH,
                    batch_size = BATCH_SIZE,
                    shuffle = True,
                    callbacks=[checkpoint, logger])

# draw and save result
y_loss = history.history['loss']
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('train_loss.png')
plt.cla()

# Test with test data set and save result
score = model.evaluate([test_left, test_right], test_disp, verbose=0)

f = open("test_result.txt", 'w')
print('Test loss:', score, file = f)
print('', file = f)
f.close()