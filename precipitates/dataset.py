import tensorflow as tf
import precipitates.precipitate as precipitate
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import itertools
import precipitates.img_tools
from tensorflow.keras import layers
import random
import os
import logging

logger = logging.getLogger("prec")

def prepare_datasets(
    train_data_root,
    crop_stride = 8,
    crop_shape=(128,128),
    batch_size = 32,
    seed = 123,
    validation_split_factor = .2,
    filter_small = False):
    
    
    img_paths = list(train_data_root.rglob("img.png"))
    mask_paths = list(train_data_root.rglob("mask.png"))
    logger.debug(f"Found {len(img_paths)} images")
    assert len(img_paths) != 0 and len(mask_paths) != 0, "Empty dataset"
    assert len(img_paths) == len(mask_paths), "number of masks is not equal to number of images"
 
    augumented_dataset_len,dataset = _get_crops_dataset(
        img_paths,
        mask_paths,
        crop_stride=crop_stride,
        crop_shape=crop_shape ,
        generator=True,
        filter_small = filter_small)
    
    cache_path= pathlib.Path('.')
    for f in cache_path.rglob('.cache*'):
        try:
            os.remove(f)
        except Exception as e :
            logger.warn(f"deleting cache: {e}")

    augument = _get_augumentation(seed=seed)
    dataset = (dataset
               .cache('.cache')
               .shuffle(batch_size)
               .map(augument,num_parallel_calls=tf.data.AUTOTUNE)
               .map(_split_imgmask,num_parallel_calls=tf.data.AUTOTUNE)
    )
    #size cropped to batch size
    train_size = int(((1-validation_split_factor) * augumented_dataset_len)//batch_size * batch_size)
    val_size = int((augumented_dataset_len - train_size)//32 *32)
    
    steps_per_epoch = train_size//batch_size 
    train_ds = dataset.take(train_size).batch(batch_size,drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    val_ds = dataset.skip(train_size).batch(batch_size,drop_remainder=True).prefetch(tf.data.AUTOTUNE)
 
    logger.debug(f"Sizes. Train: {train_size//batch_size}, Val: {val_size//batch_size}. Batch: {batch_size}")
    
    #hack force this to cache
    print(len([_ for _ in val_ds]))
    print(len([_ for _ in val_ds]))

    return train_ds,val_ds,steps_per_epoch

def img2crops(img, stride, shape):
    slider = np.lib.stride_tricks.sliding_window_view(img,shape)
    strider =  slider[::stride,::stride]
    return strider.reshape((-1,*shape))[:,:,:]


    
def get_crops_iterator(img_paths,stride, shape = (128,128),filter_small = False):
    imgs = map(precipitate.load_microscope_img,img_paths)
    
    if filter_small:
        imgs = map(precipitates.img_tools._filter_small, imgs)
    
    crop_sets_it = (img2crops(img.astype(np.float32),stride, shape) for img in imgs)
    for it in itertools.chain( crop_sets_it):
        yield from it

def _estimate_img_crops(img_shape,crop_shape,stride):
    
    h,w = img_shape
    ch,cw = crop_shape
    assert ch <= h and cw <= w, f"Image is smaller than crop f{(w,h)} vs {(cw,ch)}"
    
    if stride <= ch:
        h_steps=(h-ch)//stride +1
    else:
        h_steps=(h)//stride +1
        
    crop_per_size = []
    for c,s in [(ch,h),(cw,w)]:
        steps = 1 + (s-c)//stride
        crop_per_size.append(steps)
        
    h_steps,w_steps = crop_per_size
    return h_steps*w_steps
    
def _estimate_dataset_size(img_paths,stride, crop_shape):
    imgs = map(precipitate.load_microscope_img,img_paths)
    ch,cw = crop_shape
    
    return sum([_estimate_img_crops(img.shape, crop_shape,stride) for img in imgs])


def _get_crops_dataset(
    img_paths,
    mask_paths,
    crop_stride = 128,
    crop_shape= (128,128),
    generator=False,
    filter_small = False):
    
    img_paths = list(img_paths)
    
    tensor_shape = (crop_shape[0],crop_shape[1],4)
    img_crops_it =  get_crops_iterator(img_paths,crop_stride,crop_shape)
    mask_crops_it = get_crops_iterator(
        mask_paths,
        crop_stride,
        crop_shape,
        filter_small = filter_small
    )
    
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
    mask = tf.expand_dims(imgmask[:,:,-1],axis=2)
    return (img,mask)

def _get_augumentation(seed=123):
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical",seed=seed),
        layers.RandomRotation(0.5,seed=seed),
        layers.RandomZoom(height_factor=(-.05,.3),width_factor=(-.05,.3),seed=seed),
    ])
