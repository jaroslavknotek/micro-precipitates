---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: computer-vision
    language: python
    name: .venv
---

```python
%load_ext autoreload
%autoreload 2
import sys
sys.path.insert(0,"../python")
import nn
```

```python
import imageio
import tensorflow as tf
import pathlib
import numpy as np
import itertools
import pandas as pd
import cv2

def _filter_small(img):
    kernel = np.zeros((4,4),dtype=np.uint8)
    kernel[1:3,:]=1
    kernel[:,1:3]=1
    
    return cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)

def iou(a,b):
    a = _filter_small(imageio.imread(a))
    b = _filter_small(imageio.imread(b))
    
    m = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])
    m.update_state(a/255>.5,b/255>.5)
    return m.result().numpy()
    
def calc_similarity(arr):

    cartesian =  [[x0, y0] for x0 in arr for y0 in arr]
    rows = [(a.parent.name,b.parent.name, iou(a,b)) for a,b in cartesian]
    
    return pd.DataFrame(rows,columns=['a','b','iou'])

    
    
    
masks = list(pathlib.Path('../data/test').rglob("mask.png"))
masks = np.array(list(sorted(masks,key=lambda x: list(reversed(x.parent.name)))))
masks_2d= masks.reshape((-1,2))


df_10 = calc_similarity(masks_2d[:,0])
df_09 = calc_similarity(masks_2d[:,1])
```

```python
df_10.pivot('a',columns='b',values='iou')
```

```python
df_09.pivot('a',columns='b',values='iou')
```

```python
df_all = pd.concat([df_09,df_10])

arr =  df_all['iou']
"Average IoU: ", np.mean(arr[arr!=1])
```

```python
import app
import precipitates
import visualization

in_masks  = [m for m in masks if m.parent.name.endswith('IN')]
dfs = []
px2ums = [px/um for px,um in [(192,20),(256,50)]]
for mask_path,px2um in zip(in_masks,px2ums):
    img_out_dir = mask_path.parent
    mask_raw = imageio.imread(mask_path)
    img = precipitates.load_microscope_img(mask_path.parent/'img.png')
    mask = _filter_small(mask_raw)
    #mask = (mask_raw)
    shapes = precipitates.identify_precipitates_from_mask(mask)    
    df_features = app._get_feature_dataset(shapes)
    app._add_micrometer_scale(df_features,px2um)
    dfs.append(df_features)
    
    
    img_out_dir = mask_path.parent
    df_features.to_csv(
        img_out_dir/"precipitates.csv",
        index=False,
        header=True)
    
    fig_hist = visualization.plot_histograms(df_features)
    fig_hist.savefig(img_out_dir/"area_hist.pdf")
    fig_hist.clear()
    plt.close(fig_hist)
    
    fig_details = visualization.plot_precipitate_details(df_features,preds_mask,img)
    fig_details.savefig(img_out_dir/"precipitate_details.pdf")
    plt.close()


```
