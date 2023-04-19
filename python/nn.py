import os
import sys
import random
import warnings

import numpy as np
from tqdm.auto import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model
from keras.layers import Input,Flatten,RepeatVector,Reshape
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import keras.models
import tensorflow as tf

import dataset
warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

def predict(model, img, img_size=128, prediction_threshold = .5):
    test_data_2d = _cut_to_pieces(img,img_size)
    square_size = test_data_2d.shape[0] 
    test_data = np.array(list(map(dataset._ensure_three_chanels, test_data_2d.reshape((-1,img_size,img_size)))))
    preds_test = model.predict(test_data, verbose=0)
    preds_test = preds_test[...,0]
    preds_mask  =_decut_mask(preds_test,square_size)
    return (preds_mask> prediction_threshold).astype(np.uint8)*255

def compose_unet(crop_shape):
    assert len(crop_shape) ==2
    
    # Build U-Net model
    inputs = Input((crop_shape[0],crop_shape[1],3))
    
    s = Lambda(lambda x: x)(inputs)
    
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=[tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])],
        run_eagerly = True)
    return model

def _train_model(model,X_train,y_train,model_path,validation_split = .2):
    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint(model_path, verbose=1, save_best_only=True)
    results = model.fit(
        X_train, 
        y_train, 
        validation_split=validation_split, 
        batch_size=16, 
        epochs=50,
        callbacks=[earlystopper, checkpointer])
    
def train(img_size,channels,model_path, X_train, y_train):
    model = compose_unet(img_size,img_size,channels)
    _train_model(model, X_train,y_train, str(model_path))
    return load_model(model_path)

def load_model(model_path):
    return keras.models.load_model(str(model_path))
    
def _cut_to_pieces(img,shape):
    assert shape % 4 ==0,"Shape must be divisible by 4"
    stride = shape//4
    shape = (shape,shape)
    windows = np.lib.stride_tricks.sliding_window_view(img,shape)
    return windows[::stride,::stride]
   

def _decut_mask(crops,square_size):
    window_size = crops[0].shape[0]
    sh = window_size//4
    img_shape = (square_size -1) * sh + window_size
    windows = crops.reshape( (square_size,square_size,window_size,window_size))

    sums = np.zeros((img_shape,img_shape))
    counts = np.zeros((img_shape,img_shape))

    for i in range(square_size):
        for j in range(square_size):
            t = i*sh
            b = t + window_size
            l = j*sh
            r = l + window_size
            sums[t:b,l:r] += windows[i,j]
            counts[t:b,l:r] += 1

    return  sums/counts 



