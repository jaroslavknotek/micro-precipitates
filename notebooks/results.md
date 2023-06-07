---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: palivo
    language: python
    name: palivo
---

```python
%load_ext autoreload
%autoreload 2
```

```python
import pathlib
import pandas as pd
import json
from tqdm.auto import tqdm

def _read_line_file(file):
    res = json.load(open(file))
    
    iou = res['-1']['iou']
    precision = res['-1']['precision']
    recall = res['-1']['recall']
    f1 = res['-1']['f1']
    
    experiment = file.parent.name
    test_id,epoch = [ int(x) for x in file.stem.replace("-checkpoint","").split('_')[1:]]
    is_filter = file.parent.name.startswith('filter')
    dataset = file.parent.parent.name
    res = {
        "iou":iou,
        "f1":f1,
        "precision":precision,
        "recall":recall,
        "experiment":experiment,
        "dataset":dataset,
        "test_id":test_id,
        "epoch":epoch,
        "img_path":str(file).replace("json","png")
    }
    
    params = json.load(open(file.parent/"params.txt"))
    
    for k,v in params.items():
        res[k] = v
    
    return res
        
    

files = pathlib.Path("../tmp/20230501").rglob("test_*.json")
df = pd.DataFrame([_read_line_file(file) for file in tqdm(files,desc = 'read file')])
```

```python
import matplotlib.pyplot as plt

df_last_epochs =  df.loc[df.groupby(by=['dataset','experiment','test_id'])['epoch'].idxmax()]
df_last_epochs = df_last_epochs.sort_values(by='f1',ascending=False)
```

```python
import imageio
```

```python
import itertools
import numpy as np

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
    

# weights_dict = {99: {99: 0.9733376}, 4: {7: 0.9050117}, 5: {9: 0.6749938}, 6: {11: 0.7407341}, 7: {12: 0.52082235}, 8: {15: 0.871078}, 9: {16: 0.8958286}, 10: {17: 0.85245043}, 11: {18: 0.76666}, 12: {19: 0.54544026}, 13: {20: 0.7499881}, 14: {21: 0.9181775}, 15: {22: 0.61904}, 16: {23: 0.5789321}, 17: {24: 0.8452319}, 18: {25: 0.5499914}, 19: {26: 0.7968688}, 21: {32: 0.76040566}, 22: {30: 0.5833238}, 23: {33: 0.85999334}, 24: {31: 0.5185061}, 26: {34: 0.8055489}, 28: {35: 0.8271471}, 29: {36: 0.9166585}, 31: {37: 0.7399938}, 36: {39: 0.9415759}, 37: {40: 0.50721395}, 38: {40: 0.77081764}, 40: {41: 0.9677539}, 41: {43: 0.97035}, 42: {45: 0.86764276}, 43: {46: 0.80262446}, 44: {47: 0.84090245}, 45: {49: 0.8243181}, 46: {48: 0.56520784}, 47: {48: 0.5238}, 48: {50: 0.86289513}, 49: {51: 0.5576813}, 50: {52: 0.9238301}, 51: {53: 0.9344224}, 52: {56: 0.86903715}, 53: {55: 0.7571348}, 55: {58: 0.54346824}, 56: {59: 0.85801375}, 57: {61: 0.93103254}, 58: {62: 0.908594}, 59: {63: 0.7092904}, 60: {67: 0.86883724}, 65: {69: 0.74137217}, 66: {70: 0.92104834}, 67: {71: 0.85605156}, 68: {72: 0.5762235}, 69: {73: 0.5238}, 71: {74: 0.94444156}, 73: {77: 0.8199957}, 74: {78: 0.85565674}, 75: {79: 0.9117619}, 77: {80: 0.74999285}, 78: {81: 0.53124285}, 79: {81: 0.5624933}, 80: {82: 0.6935393}}
# _ = _construct_weight_map(weights_dict)                    
```

```python
import tensorflow as tf

def _collect_pairing_weights(p_n, p_grains,l_n, l_grains):
    weights_dict = {}
    iou_metric = tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5)
    for p_grain_id in range(1,p_n):
        p_grain_mask = it._extract_grain_mask(p_grains,p_grain_id)

        intersecting_ids = np.unique(l_grains*p_grain_mask)
        intersecting_ids = intersecting_ids[intersecting_ids>0]
        
        for l_grain_id in intersecting_ids:
            l_grain_mask = it._extract_grain_mask(l_grains,l_grain_id)
            iou_metric.update_state(l_grain_mask,p_grain_mask)
            weight = 1 - iou_metric.result().numpy()
            weights_dict.setdefault(p_grain_id,{}).setdefault(l_grain_id,weight)
            iou_metric.reset_state()
    return weights_dict

```

```python
import cv2

import scipy.optimize

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
    p_n, p_grains = cv2.connectedComponents(prediction)
    l_n, l_grains = cv2.connectedComponents(label)    
    
    # pairs only #TP
    pred_items,label_items = _pair_using_linear_sum_assignment(
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
    df = pd.DataFrame(data,columns = ['pred_id','label_id'])
    return df, p_grains, l_grains

```

```python
import sys
sys.path.insert(0,'..')

import precipitates.evaluation as ev
import precipitates.nn as nn
import precipitates.img_tools as it
from tqdm.auto import tqdm

test_imgs_folder = pathlib.Path('../data/test/IN/')
test_img_mask_pairs=ev._read_test_imgs_mask_pairs(test_imgs_folder)

res_gt_pred=[]
for row in tqdm(df_last_epochs.itertuples(),desc = "Applying model",total = len(df_last_epochs)):
    model_path =  pathlib.Path(row.img_path).parent/'model.h5' 
    model = nn.compose_unet((128,128))
    model.load_weights(model_path)
    for img,gt in test_img_mask_pairs:
        img = ev._norm(img)
        pred = nn.predict(model,img)  
        if row.filter_size >0:
            gt = it._filter_small(gt,row.filter_size)
            
        gt,pred = gt//255,pred//255
        
        df,p_prec,l_prec =  match_precipitates(pred,gt)
        p,r = ev.prec_rec(df)
        iou = ev._iou(gt,pred)
        f1 = ev._f1(p,r)
        res_gt_pred.append(
            (
                img,gt,pred,iou,f1,p,r,model_path,p_prec,l_prec
            )
        )
        
df_res = pd.DataFrame(
    res_gt_pred,
    columns= ['img','mask','pred','iou','f1','precision','recall','model_path','p_prec','l_prec']
)
```

```python
plt.plot(df_res.iou,label='iou',alpha = .5)
plt.plot(df_res.precision,label='prec',alpha = .5)
plt.plot(df_res.recall,label='rec',alpha = .5)
plt.plot(df_res.f1,label='f1',alpha = .5)
plt.legend()
```

```python
def _visualize_results(df_res):
    _,axs = plt.subplots(len(df_res),3,figsize=(16,3*len(df_res)))
    for ax_row,row in zip(axs,df_res.itertuples()):

        imgs = [
            row.img,
            row.mask,
            row.pred,
        ]
        for ax,img in zip(ax_row,imgs):
            ax.imshow(img)

        title = f"IoU:{row.iou:.3f} F1:{row.f1:.3f} P:{row.precision:.3f} R:{row.recall:.3f}"
        ax_row[0].set_title(title)

        model_name = row.model_path.parent.name 
        ax_row[1].set_title(model_name)
        data_name = row.model_path.parent.parent.name
        ax_row[2].set_title(data_name)

#_visualize_results(df_res)

n = len(df_res)
df_top_ten = df_res.sort_values(by='f1',ascending=False)

best_row = df_top_ten.iloc[0]

plt.imshow(best_row['mask'])
plt.show()
plt.imshow(best_row.pred)
plt.show()

_visualize_results(df_top_ten)
```

```python
import matplotlib.pyplot as plt
import imageio
import pathlib

test_mask_paths =list(pathlib.Path("../data/test/").glob("*IN/mask.png"))
test_mask = imageio.imread(test_mask_paths[0])
plt.imshow(test_mask)
```

```python
import sys
sys.path.insert(0,'..')
import precipitates.img_tools as img_tools
import precipitates.precipitate as precipitate
bbs = img_tools.extract_component_with_bounding_boxes(test_mask)

shapes = precipitate.identify_precipitates_from_mask(test_mask)
features = [precipitate.extract_features(shape) for shape in shapes]
shape_classes = [ precipitate.classify_shape(feature) for feature in features]

df_features = pd.DataFrame(features)
df_features['shape_class'] = shape_classes
df_features
```
