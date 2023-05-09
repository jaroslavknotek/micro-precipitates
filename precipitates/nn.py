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

import precipitates.dataset as dataset

import logging

logger = logging.getLogger("pred")


warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed
  
class WeightedBinaryCrossentropy(tf.keras.losses.Loss):
    def __init__(self,weight_zero = 1, weight_one = 1):
        super().__init__()
        self.weight_zero = float(weight_zero)
        self.weight_one = float(weight_one)
    
    def call(self, true, pred):
        """
        Calculates weighted binary cross entropy. The weights are fixed.

        This can be useful for unbalanced catagories.

        Adjust the weights here depending on what is required.

        For example if there are 10x as many positive classes as negative classes,
            if you adjust weight_zero = 1.0, weight_one = 0.1, then false positives 
            will be penalize 10 times as much as false negatives.
        """

        # calculate the binary cross entropy
        bin_crossentropy = keras.backend.binary_crossentropy(true, pred)

        # apply the weights
        weights = true * self.weight_one + (1. - true) * self.weight_zero
        weighted_bin_crossentropy = weights * bin_crossentropy 

        return keras.backend.mean(weighted_bin_crossentropy)


class DynamicallyWeightedBinaryCrossentropy(tf.keras.losses.Loss):
    def call(self, true, pred):
        """
        Calculates weighted binary cross entropy. The weights are determined dynamically
        by the balance of each category. This weight is calculated for each batch.

        The weights are calculted by determining the number of 'pos' and 'neg' classes 
        in the true labels, then dividing by the number of total predictions.

        For example if there is 1 pos class, and 99 neg class, then the weights are 1/100 and 99/100.
        These weights can be applied so false negatives are weighted 99/100, while false postives are weighted
        1/100. This prevents the classifier from labeling everything negative and getting 99% accuracy.

        This can be useful for unbalanced catagories.
        """
        # get the total number of inputs
        num_pred = keras.backend.sum(keras.backend.cast(pred < 0.5, true.dtype)) + keras.backend.sum(true)

        # get weight of values in 'pos' category
        zero_weight =  keras.backend.sum(true)/ num_pred +  keras.backend.epsilon() 

        # get weight of values in 'false' category
        one_weight = keras.backend.sum(keras.backend.cast(pred < 0.5, true.dtype)) / num_pred +  keras.backend.epsilon()

        # calculate the weight vector
        weights =  (1.0 - true) * zero_weight +  true * one_weight 

        # calculate the binary cross entropy
        bin_crossentropy = keras.backend.binary_crossentropy(true, pred)

        # apply the weights
        weighted_bin_crossentropy = weights * bin_crossentropy 

        return keras.backend.mean(weighted_bin_crossentropy)
    

def predict(model, img, img_size=128, prediction_threshold = .5):
    if np.max(img) > 1:
        logger.warning(f"Predicted img has values beyond 1. normalize to 0-1")
    
    test_data_2d = _cut_to_pieces(img,img_size)
    square_size = test_data_2d.shape[0] 
    test_data = np.array(
        list(
            map(
                _ensure_three_chanels, 
                test_data_2d.reshape((-1,img_size,img_size))
            )
        )
    )
    preds_test = model.predict(test_data, verbose=0)
    preds_test = preds_test[...,0]
    preds_mask  =_decut_mask(preds_test,square_size)
    return (preds_mask> prediction_threshold).astype(np.uint8)*255



def resolve_loss(
    loss='bc',
):
    if loss =='bc':
        return tf.keras.losses.BinaryCrossentropy()
    elif loss == 'dwbc':
        return DynamicallyWeightedBinaryCrossentropy()
    elif loss.startswith('wbc'):
        weight_zero,weight_one = [int(x) for x in loss.split('-')[1:]]
        return WeightedBinaryCrossentropy(weight_zero,weight_one)
    else:
        raise Exception(f"Unrecognized loss {loss}")
    
    
def down_block(
    inputs,
    filters, 
    activation,
    kernel_initializer,
    dropout
):
    
    c = Conv2D(filters, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding='same') (inputs)
    c = Dropout(dropout) (c)
    c = Conv2D(filters, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding='same') (c)
    p = MaxPooling2D((2, 2)) (c)
    
    return p,c

def up_block(
    in_layer,
    skip_layer,
    filters,
    activation,
    kernel_initializer,
    dropout
):
    u = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same') (in_layer)
    u = concatenate([u, skip_layer])
    c = Conv2D(filters, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding='same') (u)
    c = Dropout(dropout) (c)
    return Conv2D(filters, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding='same') (c)

def build_unet(
    crop_shape,
    loss = None,
    start_filters = 16,
    depth = 4,
    activation = 'elu',
    dropout = .2,
    kernel_initializer = 'he_normal'
):
    assert len(crop_shape) ==2
    if loss is None:
        loss = resolve_loss('bc')
    # Build U-Net model
    inputs = Input((crop_shape[0],crop_shape[1],3))
    
    in_layer = inputs
    filters = start_filters
    skip_connections = []
    for _ in range(depth):
        c,s = down_block(
            in_layer,
            filters,
            activation,
            kernel_initializer,
            dropout
        )
        filters*=2
        in_layer=c
        skip_connections.append(s)
        
    middle = Conv2D(filters, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding='same') (in_layer)
    middle = Dropout(dropout) (middle)
    middle = Conv2D(filters, (3, 3), activation=activation, kernel_initializer=kernel_initializer, padding='same') (middle)

    in_layer = middle
    for skip_connection in reversed(skip_connections):
        filters = filters//2
        in_layer = up_block(
            in_layer,
            skip_connection,
            filters,
            activation,
            kernel_initializer,
            dropout
        )

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (in_layer)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        optimizer='adam', 
        loss=loss,
        metrics=[tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])],
        run_eagerly = True)
    
    return model


def compose_unet(
    crop_shape,
    loss='bc'
):
    logging.warning("Deprecated. Use build unet fn")
    loss = resolve_loss(
        loss
    )
    return build_unet(
        crop_shape,
        loss,
        start_filters = 16,
        depth = 4,
        activation = 'elu',
        dropout = .2,
        kernel_initializer = 'he_normal'
    )
        

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

def _ensure_three_chanels(img):
    if len(img.shape) != 3 or img.shape[2]!=3:
        return np.dstack([img]*3)
    return img

