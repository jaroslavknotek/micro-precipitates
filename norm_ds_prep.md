```python
%load_ext autoreload
%autoreload 2
```


```python
import precipitates.nn as nn

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

import tensorflow as tf

class ToThreeChannels(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
      
    def call(self, inputs):
        return tf.stack([inputs]*3,axis=3)

def build_unet(
    loss = None,
    start_filters = 16,
    depth = 4,
    activation = 'elu',
    dropout = .2,
    kernel_initializer = 'he_normal'
):
    if loss is None:
        loss = resolve_loss('bc')
    # Build U-Net model
    inputs = Input((None,None))
    in_layer = ToThreeChannels()(inputs)
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

def resolve_loss(*args):
    return nn.resolve_loss(*args)

def down_block(*args):
    return nn.down_block(*args)

def up_block(*args):
    return nn.up_block(*args)

model = build_unet()
```

    2023-06-18 17:29:09.362227: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory; LD_LIBRARY_PATH: /home/jry/.conda/envs/palivo/lib/python3.10/site-packages/cv2/../../lib64:
    2023-06-18 17:29:09.362243: W tensorflow/compiler/xla/stream_executor/cuda/cuda_driver.cc:265] failed call to cuInit: UNKNOWN ERROR (303)
    2023-06-18 17:29:09.362254: I tensorflow/compiler/xla/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (T14): /proc/driver/nvidia/version does not exist
    2023-06-18 17:29:09.362356: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 AVX_VNNI FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.



```python
import pathlib

weights_path = pathlib.Path('model/model-20230427192618.h5')
model.load_weights(weights_path)
```


```python

import imageio
import matplotlib.pyplot as plt
img = imageio.imread('data/test/IN/DELISA LTO_08Ch18N10T_pricny rez_nulty stav_TOP_BSE_10/img.png')
img = img/np.max(img)
mask = imageio.imread('data/test/IN/DELISA LTO_08Ch18N10T_pricny rez_nulty stav_TOP_BSE_10/mask.png')
mask = mask//np.max(mask)
plt.imshow(img)


pred_input = np.expand_dims(img,axis=0)

pred_res = np.squeeze(model.predict(pred_input))
thr = np.mean(pred_res)*5
pred_res[pred_res>thr] = 1
pred_res[pred_res<=thr] = 0
plt.show()
plt.imshow(pred_res)
plt.show()
plt.imshow(pred_res.astype(float) - mask,cmap='seismic')
```

    /tmp/ipykernel_99704/4106452430.py:3: DeprecationWarning: Starting with ImageIO v3 the behavior of this function will switch to that of iio.v3.imread. To keep the current behavior (and make this warning disappear) use `import imageio.v2 as imageio` or call `imageio.v2.imread` directly.
      img = imageio.imread('data/test/IN/DELISA LTO_08Ch18N10T_pricny rez_nulty stav_TOP_BSE_10/img.png')
    /tmp/ipykernel_99704/4106452430.py:5: DeprecationWarning: Starting with ImageIO v3 the behavior of this function will switch to that of iio.v3.imread. To keep the current behavior (and make this warning disappear) use `import imageio.v2 as imageio` or call `imageio.v2.imread` directly.
      mask = imageio.imread('data/test/IN/DELISA LTO_08Ch18N10T_pricny rez_nulty stav_TOP_BSE_10/mask.png')


    1/1 [==============================] - 1s 571ms/step



    
![png](output_3_2.png)
    



    
![png](output_3_3.png)
    





    <matplotlib.image.AxesImage at 0x7f99d42d9fc0>




    
![png](output_3_5.png)
    



```python
import precipitates.dataset as dataset

dataset_root = pathlib.Path('data/20230617-normalized/labeled/')
ds_train,ds_val = dataset.prepare_datasets(dataset_root,crop_size=64,repeat=50,filter_size = 20)
```


```python
import itertools

for (i1,m1),(i2,m2) in itertools.islice(zip(ds_val,ds_val),0,10):
    _,(axl,axlm,axr,axrm) = plt.subplots(1,4)
    axl.imshow(i1[0])
    axlm.imshow(m1[0])
    axr.imshow(i2[0])
    axrm.imshow(m2[0])
    plt.show()
```


    
![png](output_5_0.png)
    



    
![png](output_5_1.png)
    



    
![png](output_5_2.png)
    



    
![png](output_5_3.png)
    



    
![png](output_5_4.png)
    



    
![png](output_5_5.png)
    



    
![png](output_5_6.png)
    



    
![png](output_5_7.png)
    



    
![png](output_5_8.png)
    



    
![png](output_5_9.png)
    



```python

```
