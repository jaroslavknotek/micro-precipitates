import tensorflow as tf
import precipitates.precipitate as precipitate
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import itertools
import precipitates.img_tools as img_tools
from tensorflow.keras import layers
import random
import os
import logging
import albumentations as A
import cv2
import imageio

logger = logging.getLogger("prec")

def prepare_datasets(
    dataset_root,
    crop_size=128,
    batch_size = 32,
    seed = 123,
    validation_split_factor = .2,
    filter_size = 0,
    repeat = 100,
    interpolation=cv2.INTER_CUBIC):

    logger.info(f"Loading Dataset from {dataset_root}")
    
    np.random.seed = seed
    dataset_array = load_img_mask_pair(dataset_root,filter_size)
    train_size,val_size = _get_train_test_size(len(dataset_array), validation_split_factor)
    
    ds_images = tf.data.experimental.from_list(dataset_array)
    
    dss = [ds_images.take(train_size),ds_images.skip(train_size)]
    return [_prep_ds(ds,repeat,batch_size,crop_size,interpolation) for ds in dss]


def load_img_mask_pair(dataset_root,filter_size = 0):
    dataset_array = _load_pairs(dataset_root)
    return [(img,img_tools.filter_small(mask,filter_size)) for img,mask in dataset_array]


def _get_train_test_size(n,validation_split_factor):
    
    val_size = int(np.ceil(n*validation_split_factor))
    train_size = n-val_size
    assert n>0
    assert val_size >0
    assert train_size >0
    
    return train_size,val_size
    


def _get_augumentation(crop_size,interpolation):
    return A.Compose([
        A.PadIfNeeded(crop_size,crop_size),
        A.RandomCrop(crop_size,crop_size),
        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.augmentations.transforms.GaussNoise(.2),
        A.Rotate(limit=45, interpolation=interpolation),
        A.ElasticTransform(p=.5,
                           alpha=50, 
                           sigma=120 * 0.05,
                           alpha_affine=120 * 0.03,
                           interpolation=interpolation
                          )
    ])
    

def _prep_ds(ds,repeat,batch_size,crop_size,interpolation):
    
    transform = _get_augumentation(crop_size,interpolation)
    def aug(image,mask):
        transformed = transform(image=image ,mask=mask)
        return transformed['image'],transformed['mask']
    @tf.function
    def process_data(image,mask):
        return  tf.numpy_function(aug,inp=[image,mask], Tout = (tf.float32,tf.float32))

    return ds.repeat(repeat)\
        .map(process_data,num_parallel_calls=tf.data.experimental.AUTOTUNE)\
        .batch(batch_size,drop_remainder=True)\
        .prefetch(tf.data.AUTOTUNE)

def _norm_float(img):
    img_f = np.float32(img)
    img_min = np.min(img_f)
    img_max = np.max(img_f)
    return (img_f-img_min)/(img_max-img_min)
    

def _ensure_2d(img):
    match img.shape:
        case (h,w):
            return img
        case (h,w,_):
            return img[:,:,0]

def load_image(img_path,normalize=True,ensure_two_channels = True,ensure_square = True):    
    img = imageio.imread(img_path)
    
    if ensure_square:
        m = np.min(list(img.shape[:2]))
        img = img[:m,:m]
    
    if ensure_two_channels:
        img = _ensure_2d(img)
    
    if normalize:
        img = _norm_float(img)
    
    return img

def _get_img_mask_iter(dataset_root):
    dataset_root = pathlib.Path(dataset_root)
    for img_root in dataset_root.glob('*'):
        try:
            img = load_image(img_root/'img.png')
            mask = load_image(img_root/'mask.png')
            mask = (mask>0).astype(mask.dtype)
            yield img,mask
        except FileNotFoundError as e:
            logger.warning(f"Skipped {img_root}. Didn't find both files. Error {e}")
    
def _load_pairs(dataset_root):
    return list(_get_img_mask_iter(dataset_root))