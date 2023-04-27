---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: cv
    language: python
    name: .venv
---

```python
%load_ext autoreload
%autoreload 2
```

```python
%cd ../../micro-precipitates/
```

```python
import pathlib
models_tmp = list(pathlib.Path('tmp').rglob("*.h5"))
models_dir = list(pathlib.Path('model').rglob("*.h5"))
all_models = models_tmp + models_dir
```

```python
# /tmp/20230427152953/model.h5

```

```python
import matplotlib.pyplot as plt
import imageio
test_data_dirs = list( pathlib.Path('data/test').rglob('*IN/img.png'))

for img_path in test_data_dirs:
    mask_path = img_path.parent/'mask.png'
    img = imageio.imread(img_path)
    mask =imageio.imread(mask_path)
```

```python

        
```

```python
import precipitates.img_tools as it
import numpy as n

def _pair_grains(*params):
    return it._pair_grains(*params)

# def compare(predicted,label,include_df = False):
#     df =  _pair_grains(predicted,label)
    
#     grains_pred = df['pred_id'].max()
#     grains_label = df['label_id'].max()

#     # todo check that pairs are not twice
#     tp = len(df[ ~df['label_id'].isna() & ~df['pred_id'].isna()])
#     fp = len(df[ df['label_id'].isna() & ~df['pred_id'].isna()])
#     fn = len(df[ ~df['label_id'].isna() & df['pred_id'].isna()])
#     tn = 0 #len(df[ ~df['label_id'].isna() & df['pred_id'].isna()])

#     precision = np.sum(~df['label_id'].isna() & ~df['pred_id'].isna()) / grains_pred
    
#     recall =  np.sum(~df['label_id'].isna() & ~df['pred_id'].isna()) / grains_label
    
    
#     if not include_df:
#         return precision,recall
#     return precision,recall,df


# _,mask,pred = img_rows[0]

# print(compare(pred,mask))
# plt.imshow(mask)
# plt.show()
# plt.imshow(pred)
```

```python
def _intersection(a,b):
    if a is None or b is None:
        return np.nan
    else:
        return a*b
    
def _union(a,b):
    assert a is not None or b is not None
        
    if a is None:
        a = np.zeros(b.shape)
    if b is None:
        b = np.zeros(a.shape)
    
    return np.sum(a+b>=1)


```

```python
import itertools
def _category(row,clusters):
    min_size = np.nanmin([row.label_area_px,row.pred_area_px])
    for a,b in itertools.pairwise(clusters):
        if a<=min_size<b:
            return clusters.index(a)
    assert False

```

```python

def prec_rec(df):
    grains_pred = df['pred_id'].max()
    grains_label = df['label_id'].max()

    # todo check that pairs are not twice
    tp = len(df[ ~df['label_id'].isna() & ~df['pred_id'].isna()])
    fp = len(df[ df['label_id'].isna() & ~df['pred_id'].isna()])
    fn = len(df[ ~df['label_id'].isna() & df['pred_id'].isna()])
    tn = 0 #len(df[ ~df['label_id'].isna() & df['pred_id'].isna()])

    precision = np.sum(~df['label_id'].isna() & ~df['pred_id'].isna()) / grains_pred
    
    recall =  np.sum(~df['label_id'].isna() & ~df['pred_id'].isna()) / grains_label
    
    return precision,recall

```

```python
import tensorflow as tf
import json

def _merge(masks):
    masks = [mask for mask in masks if mask is not None]
    return np.uint8(np.sum(masks,axis=0)>0)
   
def _iou(label,pred):
    m = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])
    m.update_state(label,pred)
    return m.result().numpy()

def _iou_from_arr(label_arr,pred_arr):
    pred_mask = _merge(label_arr)
    label_mask = _merge(pred_arr)
    if pred_mask.shape != label_mask.shape or len(label_mask) !=2:
        return np.nan
    return _iou(label_mask,pred_mask)

def _f1(precision, recall):
    return 2*(precision * recall)/(precision + recall)

def _calculate_metrics(pred,label,clusters = [0,50,500,1024**2]):
    df =  _pair_grains(pred,label)
    
    df['pred_area_px'] = [ np.sum(x) for x in df.pred_mask]
    df['label_area_px'] = [ np.sum(x) for x in df.label_mask]
    df['size_category'] = [ _category(row,clusters) for row in df.itertuples()]
    
    cat_size = {u:df[df.size_category == u] for u in df.size_category.unique()}
    
    cat_size[-1] = df
    
    metrics = {}
    for u,sc_df in cat_size.items():
        iou =_iou_from_arr(
            sc_df.pred_mask.to_numpy(),
            sc_df.label_mask.to_numpy()
        )
        p,r = prec_rec(sc_df)
        metrics[int(u)] = {
            "iou":float(iou),
            "precision":float(p),
            "recall":float(r),
            "f1":float(_f1(p,r))
        }
    # force delete
    df = None
    del df
    del cat_size
    return metrics

```

```python
# df =  _pair_grains(pred,mask)
# df['pred_area_px'] = [ np.sum(x) for x in df.pred_mask]
# df['label_area_px'] = [ np.sum(x) for x in df.label_mask]
# df['intersection'] = [ _intersection(a,b) for a,b in zip(df.label_mask,df.pred_mask)]
# df['union'] = [ _union(a,b) for a,b in zip(df.label_mask,df.pred_mask)]
# df['iou' ] = df['intersection']/df['union']
# clusters = [0,50,500,1024**2]
# df['size_category'] = [ _category(row,clusters) for row in df.itertuples()]
# plt.hist(df['label_area_px'],bins=100)
# plt.hist(df['pred_area_px'],bins=100)
# plt.show()
# plt.hist(df['size_category'],bins=5)
# metrics_res = _calculate_metrics(pred,mask)
# print(json.dumps(metrics_res,indent=True))
    
```

```python
import precipitates.nn as nn
from tqdm.auto import tqdm
import numpy as np
import pathlib
from tqdm.auto import tqdm


def _norm(img):
    img_min=np.min(img)
    img_max = np.max(img)
    return (img.astype(float)-img_min) / (img_max-img_min)

def _get_metrics_with_img():
    results = []
    
    for model_path in tqdm(all_models):
        model = nn.compose_unet((128,128))
        model.load_weights(model_path)

        for img_path in test_data_dirs:

            mask_path = img_path.parent/'mask.png'
            img = imageio.imread(img_path)
            mask =imageio.imread(mask_path)
            #pred = nn.predict(model,img.astype(float))

            pred = nn.predict(model,_norm(img))    
            metrics_res = _calculate_metrics(pred,mask)
            results.append((img,mask,pred,metrics_res))
        
    return results

results = _get_metrics_with_img()
```

```python
mods = np.dstack([all_models]*2).flatten()
_,axs = plt.subplots(len(results),4,figsize=(16,4*len(results)))
for ax_r,row,model_path in zip(axs,results,mods):
    img,mask,pred,metrics_res = row
    
    f1 = metrics_res[-1]['f1']
    iou = metrics_res[-1]['iou']
    
    title = f"{model_path.stem} - f1:{f1},iou:{iou}"
    ax_r[1].set_title(title)
    for ax,img in zip(ax_r,row[:-1]):
        ax.imshow(img)
        
    idxs = list(metrics_res.keys())
    ious = [v['iou'] for v in metrics_res.values()]
    f1s = [v['f1'] for v in metrics_res.values()]
    ax_r[-1].plot(idxs,f1s,'x',label='F1')
    ax_r[-1].plot(idxs,ious,'x',label='IOU')
    ax_r[-1].legend()
    
```

```python
assert False
```

```python
import pathlib
import shutil

data_dir = pathlib.Path('data')


labeled_paths =  [f for f in data_dir.glob("labeled/*")]
labeled_names = set([f.name for f in labeled_paths])

all_paths  = [f for f in data_dir.rglob("**/*.*") if f.suffix in {'.tif'} and f.stem not in {'img','mask'} ]
all_paths_w_names = [(f.stem.replace(' ','_'),f) for f in all_paths]

not_labeled = [ path for name,path in all_paths_w_names if name not in labeled_names]
(data_dir/'not_labeled').mkdir(exist_ok=True)
for f in not_labeled:
    
    shutil.copy(f,data_dir/'not_labeled'/f.name)
```

```python
import sys
sys.path.insert(0,'../python')

import nn
import pathlib
```

```python
model_path = pathlib.Path('/home/jry/source/jaroslavknotek/micro-precipitates/tmp/20230415/model.h5')
model = nn.compose_unet(128,128,3)
model.load_weights(model_path)
```

```python
import imageio
import precipitates
import matplotlib.pyplot as plt

test_data_dirs = list( pathlib.Path('../data/test').glob('*'))

import tensorflow as tf

test_dir = test_data_dirs[0]
#for test_dir in test_data_dirs:
img = precipitates.load_microscope_img(test_dir/'img.png')
mask = imageio.imread(test_dir/'mask.png')
assert img.shape == (1024,1024)

_,axs = plt.subplots(1,2)
axs[0].imshow(img)
predicted = nn.predict(model,img)
```
