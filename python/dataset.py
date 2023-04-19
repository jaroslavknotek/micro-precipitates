import tensorflow as tf
import precipitates
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import itertools

from tensorflow.keras import layers
import random
import os 

def prepare_datasets(
    train_data_root,
    crop_stride = 8,
    crop_shape=(128,128),
    batch_size = 32,
    repeat_data = 2,
    seed = 123,
    validation_split_factor = .2):
    
    img_paths = list(train_data_root.rglob("img.png"))
    mask_paths = list(train_data_root.rglob("mask.png"))
    assert len(img_paths) != 0 and len(mask_paths) != 0, "Empty dataset"
    assert len(img_paths) == len(mask_paths), "number of masks is not equal to number of images"
    
    ds_len,dataset = _get_crops_dataset(img_paths,mask_paths,crop_stride=crop_stride,crop_shape=crop_shape ,generator=True)
    augument = _get_augumentation(seed=seed)
    augumented = dataset.repeat(repeat_data).map(augument,num_parallel_calls=tf.data.AUTOTUNE).shuffle(200,seed=seed).map(_split_imgmask,num_parallel_calls=tf.data.AUTOTUNE)

    augumented_dataset_len = ds_len*repeat_data
    validation_size = int(augumented_dataset_len * validation_split_factor)
    val_ds = augumented.take(validation_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    train_ds = augumented.skip(validation_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    steps_per_epoch = (augumented_dataset_len-validation_size) // batch_size
    return train_ds,val_ds,steps_per_epoch


def img2crops(img, stride, shape):
    slider = np.lib.stride_tricks.sliding_window_view(img,shape)
    strider =  slider[::stride,::stride]
    return strider.reshape((-1,*shape))[:,:,:]


    
def get_crops_iterator(img_paths,stride, shape = (128,128)):
    imgs = map(precipitates.load_microscope_img,img_paths)
    
    crop_sets_it = (img2crops(img.astype(np.float32),stride, shape) for img in imgs)
    for it in itertools.chain( crop_sets_it):
        yield from it

def _estimate_dataset_size(img_paths,stride, shape):
    imgs = map(precipitates.load_microscope_img,img_paths)
    
    total =0
    for img in imgs:
        ch,cw = shape
        h,w = img.shape
        h_steps=(h-ch)/stride +1
        w_steps=(w-cw)/stride +1
        crops = h_steps*w_steps
        total += int(crops)
    return total


def _get_crops_dataset(
    img_paths,
    mask_paths,
    crop_stride = 128,
    crop_shape= (128,128),
    generator=False):
    img_paths = list(img_paths)
    
    tensor_shape = (crop_shape[0],crop_shape[1],4)
    img_crops_it =  get_crops_iterator(img_paths,crop_stride,crop_shape)
    mask_crops_it = get_crops_iterator(mask_paths,crop_stride,crop_shape)
    
    three_channels =  (np.dstack([i,i,i,m])/255 for i,m in zip(img_crops_it,mask_crops_it))
    if generator:
        dataset_size = _estimate_dataset_size(img_paths,crop_stride,crop_shape)
        return dataset_size, tf.data.Dataset.from_generator(
            lambda: three_channels ,
            output_signature=tf.TensorSpec(shape=tensor_shape))
    else:
        arr = np.fromiter( three_channels ,dtype=(np.float32,tensor_shape))
        dataset_size = arr.shape[0]
        return dataset_size, tf.data.Dataset.from_tensor_slices(arr[...])

    
def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.random.set_seed(seed)
    np.random.seed = (seed)
    random.seed = seed

@tf.function
def _split_imgmask(imgmask):
    img = imgmask[:,:,:3]
    mask = imgmask[:,:,-1]
    return (img,mask)

def _get_augumentation(seed=123):
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical",seed=seed),
        layers.RandomRotation(0.5,seed=seed),
        layers.RandomZoom(height_factor=(-.05,.3),width_factor=(-.05,.3),seed=seed),
    ])
