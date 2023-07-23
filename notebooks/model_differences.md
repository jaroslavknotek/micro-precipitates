---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.6
  kernelspec:
    display_name: palivo_general
    language: python
    name: palivo_general
---

```python
%load_ext autoreload
%autoreload 2
%cd ../../micro-precipitates

import sys
sys.path.insert(0,'precipitates')
import precipitates.nn as nn
```

# Load Models

```python
import pathlib

# small models
model_paths = [
    '../tmp/20230625130043_20230623_loss=bfl-filter_size=0-cnn_filters=8-cnn_depth=4-cnn_activation=elu-crop_size=128.h5',
    '../tmp/20230625133057_20230623_loss=bfl-filter_size=0-cnn_filters=8-cnn_depth=3-cnn_activation=relu-crop_size=128.h5',
    '../tmp/20230625124700_20230623_loss=bfl-filter_size=0-cnn_filters=8-cnn_depth=3-cnn_activation=elu-crop_size=64.h5'
]

# large models
model_paths =[
    '../tmp/20230625210950_20230623_loss=bfl-filter_size=0-cnn_filters=8-cnn_depth=5-cnn_activation=elu-crop_size=256.h5',
    '../tmp/20230625214709_20230623_loss=bfl-filter_size=0-cnn_filters=8-cnn_depth=6-cnn_activation=elu-crop_size=256.h5',
    '../tmp/20230625222548_20230623_loss=bfl-filter_size=0-cnn_filters=8-cnn_depth=7-cnn_activation=elu-crop_size=512.h5'
]

model_paths = list(map(pathlib.Path, model_paths))


def param_dict_from_param_string(param_string):
    key_vals = param_string.split('-')
    pairs = [kv.split('=') for kv in key_vals]
    return dict(pairs)
    
    
prefix_len = 24
param_strings = [model_path.stem[24:] for model_path in model_paths]
param_dicts = list(map(param_dict_from_param_string, param_strings))


def _build_from_param_dict(model_path, param_dict):
    model = nn.build_unet(
        int(param_dict['crop_size']),
        start_filters = int(param_dict['cnn_filters']),
        depth = int(param_dict['cnn_depth']),
        activation = param_dict['cnn_activation']
    )
    model.load_weights(model_path)
    return model

models = [ _build_from_param_dict(p,d) for p,d in zip(model_paths,param_dicts)]
THR = .5
```

## Ensemble

```python
data_root = pathlib.Path('data/20230623/labeled')
train_ds = list(ds.load_img_mask_pair(data_root,filter_size = 10))

train_res = []
for img,gt in tqdm(train_ds,desc='Data'):
    res_row = []
    try:
        for model in tqdm(models,desc='Models'):
            pred = nn.predict(model,img,return_raw=True)
            res_row.append((img,gt,pred))
        train_res.append(res_row)
    except Exception as e:
        print("error",img.shape,e)
           
```

```python
n_models = len(train_res[0])

per_model = {}

for train_row in train_res:
    for i in range(n_models):
        model_res = per_model.setdefault(i,[])
        _,gt,pred = train_row[i]
        model_res.append(pred)
    per_model.setdefault('gt',[]).append(gt)
```

```python
import numpy as np


n = 20
weights = np.random.random_sample(size=(n,n_models))
```

```python
weights_results =[]
for n_model_weights in tqdm(weights):
    preds = [per_model[i] for i in range(len(n_model_weights))]
    for imgs,gt in zip(zip(*preds),per_model['gt']):
        weighted_raw = np.sum(np.array(imgs).T * n_model_weights,axis=2).T
        weights_results.append([
            n_model_weights,
            weighted_raw,
            gt
        ])
```

```python

```

```python
import precipitates.evaluation as ev


metrics = []
for w,pred,gt in tqdm(weights_results):
    rr = ev.calculate_metrics(np.uint8(pred>THR),np.uint8(gt>0),component_limit = 1500) 
    metrics.append(rr)
```

```python
f1s = np.array([m[-1]['f1'] for m in metrics])

f1s_per_ws = f1s.reshape((len(weights),-1))

f1ws = np.array([wr[0] for wr in weights_results])
xxx=f1ws.reshape((len(weights),-1,3))

f1_sort= np.argsort(np.nanmean(f1s_per_ws,axis=1))[::-1]

for i,(f,xx) in enumerate(zip(f1s_per_ws[f1_sort],xxx[f1_sort])):
    plt.plot(f,label=i)
    print(np.nanmean(f),np.nanvar(f), xx[0],np.sum(xx[0]))
plt.legend()
```

```python

sample_id = 2
plt.imshow(train_ds[sample_id][0])
plt.show()
plt.imshow(train_ds[sample_id][1])
plt.show()

preds = np.array([p for _,p,_ in weights_results],dtype=object)
preds = preds.reshape((len(weights),-1))
y = preds[0,1]
x = weights_results[1][1]
print(np.max(x),np.max(y))
sample = preds[0][sample_id]
plt.title(np.max(sample))
plt.imshow(sample > THR)
```

```python
f1s = [m[-1]['f1'] for m in metrics]
f1s = np.sort(f1s)
plt.plot(f1s)
#plt.plot(list(sorted(f1s)))

    
```

## Measure Test

```python
import precipitates.dataset as ds
import matplotlib.pyplot as plt

test_root = pathlib.Path('data/test/')

test_ds = ds.load_img_mask_pair(test_root)
plt.imshow(test_ds[0][0])
plt.show()
plt.imshow(test_ds[0][1])

```

```python
from tqdm.auto import tqdm
import precipitates.evaluation as ev
import numpy as np

res_test = []
for img,gt in tqdm(test_ds,desc='Data'):
    res_row = []
    for model in tqdm(models,desc='Models'):
        pred = nn.predict(model,img,return_raw=True)
        rr = ev.calculate_metrics(np.uint8(pred>THR),np.uint8(gt>0),component_limit = 1500) 
        res_row.append((img,gt,pred,rr))
    res_test.append(res_row)
    

        
```

```python
h = len(res_test)
w = len(res_test[0]) +2

_,axs = plt.subplots(h,w,figsize = (8*w,8*h))
for ax_row,img_results in zip(axs,res_test):
    
    for ax,(img,gt,pred,m) in zip(ax_row[2:], img_results):
        pred_thr = pred>THR
        pred = -pred
        pred[pred_thr] = 1
        ax.set_title(m[-1]['f1'])
        ax.imshow(pred,vmin=0,vmax=1,cmap='seismic')
        
    ax_row[0].imshow(img)
    ax_row[1].imshow(gt)
    
    _=[ax.axis('off') for ax in ax_row]
        
plt.savefig('eval.pdf')    
```

## Weight Map

```python
h = len(res)
w = len(res[0]) +2

_,axs = plt.subplots(h,w,figsize = (5*w,5*h))
for ax_row,img_results in zip(axs,res):
    for ax,(img,gt,pred) in zip(ax_row[2:], img_results):
        ax.imshow(gt - pred,vmin=-1,vmax=1,cmap='seismic')
    ax_row[0].imshow(img)
    ax_row[1].imshow(gt)
        
        
```
