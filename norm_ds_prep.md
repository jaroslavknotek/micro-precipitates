```python
%load_ext autoreload
%autoreload 2
```


```python
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
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

```python
import precipitates.dataset as dataset
import pathlib

dataset_root = pathlib.Path('data/20230617-normalized/labeled/')
ds_train,ds_val = dataset.prepare_datasets(dataset_root,crop_size=128,repeat=100,filter_size = 16)
```


```python
import itertools
import matplotlib.pyplot as plt

for (i1,m1),(i2,m2) in itertools.islice(zip(ds_val,ds_val),0,10):
    _,(axl,axlm,axr,axrm) = plt.subplots(1,4)
    axl.imshow(i1[0])
    axlm.imshow(m1[0])
    axr.imshow(i2[0])
    axrm.imshow(m2[0])
    plt.show()
```
```python
import cv2
import scipy.optimize

def _construct_weight_map(weights_dict):
    # Remap arbitrary indices to integers
    p_map= {}

    for i,v in enumerate(weights_dict.keys()):
        p_map[v]=i

    l_keys = itertools.chain(
                *(list(k for k in v.keys()) for v in weights_dict.values())
            )
    l_unique = np.unique(list(l_keys))
    l_map={}
    for i,v in enumerate(l_unique):
        l_map[v]=i
        
    weights = np.zeros((len(p_map),len(l_map)))
    for i,(p,pv) in enumerate(weights_dict.items()):
        for l,lv in pv.items():
            weights[p_map[p],l_map[l]] = lv
    return weights,p_map,l_map
    
def _extract_grain_mask(labels,grain_id):
    grain = labels.copy()
    grain[labels == grain_id] = 1
    grain[labels != grain_id] = 0
    
    if (grain == 0).all():
        raise Exception(f"Grain {grain_id} not found")
    
    return grain


def _collect_pairing_weights(p_n, p_grains,l_n, l_grains):
    weights_dict = {}
    iou_metric = tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5)
    for p_grain_id in range(1,p_n):
        p_grain_mask = _extract_grain_mask(p_grains,p_grain_id)

        intersecting_ids = np.unique(l_grains*p_grain_mask)
        intersecting_ids = intersecting_ids[intersecting_ids>0]
        
        for l_grain_id in intersecting_ids:
            l_grain_mask = _extract_grain_mask(l_grains,l_grain_id)
            iou_metric.update_state(l_grain_mask,p_grain_mask)
            weight = 1 - iou_metric.result().numpy()
            weights_dict.setdefault(p_grain_id,{}).setdefault(l_grain_id,weight)
            iou_metric.reset_state()
    return weights_dict

def _pair_using_linear_sum_assignment(p_n, p_grains,l_n, l_grains, cap=500):
    
    if cap is not None:
        p_n = min(cap,p_n)
        p_grains[p_grains >cap] = 0
        
        l_n = min(cap,l_n)
        l_grains[l_grains >cap] = 0
        
    weights_dict = _collect_pairing_weights(p_n, p_grains,l_n, l_grains)
    weights,p_map,l_map = _construct_weight_map(weights_dict)
    p_item_id,l_item_id = scipy.optimize.linear_sum_assignment(weights)
    
    inverse_p_map = { v:k for k,v in p_map.items()}
    p_item = np.array([inverse_p_map[idx] for idx in p_item_id])
    inverse_l_map = { v:k for k,v in l_map.items()}
    l_item = np.array([inverse_l_map[idx] for idx in l_item_id])
    return p_item,l_item

def match_precipitates(prediction,label):
    p_n, p_grains = cv2.connectedComponents(np.uint8(prediction))
    l_n, l_grains = cv2.connectedComponents(np.uint8(label))
    
    # pairs only #TP
    pred_items,label_items =  _pair_using_linear_sum_assignment(
        p_n, 
        p_grains,
        l_n, 
        l_grains
    )
    data = list(zip(pred_items,label_items))
    #FP
    p_set = set(pred_items)
    false_positives = [ i for i in range(1,p_n) if i not in p_set]
    for i in false_positives:
        data.append((i,None))
    
    #FN
    l_set = set(label_items)
    label_positives = [i for i in range(1,l_n) if i not in l_set]
    for i in label_positives:
        data.append((None,i))
    
    fn = len([ (p,l) for p,l in data if p is not None and l is None ])
    fp = len([ (p,l) for p,l in data if l is not None and p is None ])
    tp =len([ (p,l) for p,l in data if p is not None and l is not None])
    
        
    return tp,fp,fn


```

```python
l1,l2= [m for i,m in itertools.islice(ds_train,0,2)]
```

```python
[(plt.imshow(p) and plt.show()) for p in l1.numpy()[[14,28]]]
```

```python
i1,i2 = l1.numpy()[[14,28]]
match_precipitates(i1,i2)
```

```python
class ComponentF1(tf.keras.metrics.Metric):

    def __init__(self, name='component_f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight('tp', initializer = 'zeros')
        self.fp = self.add_weight('fp', initializer = 'zeros')
        self.fn = self.add_weight('fn', initializer = 'zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        
        tp,fp,fn = match_precipitates(y_pred.numpy(),y_true.numpy())
        self.tp.assign_add(tf.constant(float(tp)))
        self.fp.assign_add(tf.constant(float(fp)))
        self.fn.assign_add(tf.constant(float(fn)))
        

    def result(self):
        precision = self.tp / (self.tp + self.fp)
        recall = self.tp / (self.tp + self.fn)
        if precision + recall ==0:
            return np.nan
        
        return 2*(precision * recall)/(precision + recall)


myf1 = ComponentF1()
i1 = l1[14]
i2 = l1[28]
myf1.update_state(i1,i2)
myf1.result()

```

```python

```
