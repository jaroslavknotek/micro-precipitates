---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.6
  kernelspec:
    display_name: torch_cv
    language: python
    name: torch_cv
---

```python
%load_ext autoreload
%autoreload 2
```

```python
# import os
# os.environ['CUDA_LAUNCH_BLOCKING']='1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'
```

```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
```

```python
import pathlib

training_output_root = pathlib.Path(f'../../training/denoiseg-SEM-precipitates/')
```

```python
import imageio
import matplotlib.pyplot as plt
import pathlib
import denoiseg.image_utils as iu



labeled_root = pathlib.Path('../data/20230921_rev/labeled/')
data_denoise_root = pathlib.Path('../data/delisa-all/')
data_test_root = pathlib.Path('../data/test/')


imgs_paths = list(labeled_root.rglob('img.png'))
imgs_lbl = [iu.load_image(img_path) for img_path in imgs_paths]

mask_paths = [ img_path.parent/"mask.png" for img_path in imgs_paths]
gts_lbl = [iu.load_image(img_path) for img_path in mask_paths]

imgs_names = set([p.parent for p in imgs_paths])
data_denoise_root = [p for p in data_denoise_root.rglob('img.png') if p.parent.name not in imgs_names]
imgs_denoise = list(map(iu.load_image,data_denoise_root))
gts_denoise = [None]*len(imgs_denoise)

imgs_test_paths = list(data_test_root.rglob('img.png'))
imgs_test = [iu.load_image(img_path) for img_path in imgs_test_paths]

mask_tests_paths = [ img_path.parent/"mask.png" for img_path in imgs_test_paths]
gts_test = [iu.load_image(img_path) for img_path in mask_tests_paths]

assert len(gts_lbl) == len(imgs_lbl)

print(f"{len(imgs_lbl)=} {len(imgs_test)=} {len(imgs_denoise)=}")
imgs = imgs_lbl + imgs_denoise
gts = gts_lbl + gts_denoise
```

```python
import numpy as np
import denoiseg.configuration as cfg
default_params = cfg.get_default_config()

custom = {
    "patch_size":128,
    "epochs":200,
    "patience":20,
    "dataset_repeat":50,
    "denoise_loss_weight":.005,
    "denoise_enabled":False,
    "validation_set_percentage":.2,
    
    "model":{
        "filters":8,
        "depth":5,
        
    },
    "augumentation":{
        "brightness_contrast":True,
        "noise_val":0.001,
        "flip_vertical": False,
        "rotate_deg": 10,
        "elastic":True,
        "blur_sharp_power": 1,
    }
}

train_params = cfg.merge(default_params,custom)
```

```python
def unet_weight_map(y, wc=None, w0 = 10, sigma = 5):
    
    labels = label(y)
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[:,:,i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=2)
        d1 = distances[:,:,0]
        d2 = distances[:,:,1]
        w = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels
        
        if wc:
            class_weights = np.zeros_like(y)
            for k, v in wc.items():
                class_weights[y == k] = v
            w = w + class_weights
        else:
            w = w +1
    else:
        w = np.zeros_like(y)
    
    return w

```

```python
import denoiseg.visualization as vis

vis.sample_ds(
    imgs,
    gts,
    train_params
)
```

# Training


- disable denoise
- different augumentations
...

```python
import denoiseg.segmentation as seg

checkpoint, out_losses =seg.run_training(
    imgs,
    gts,
    train_params, 
    training_output_root,
    device = device
)
```

```python
import denoiseg.visualization as vis

_= vis.plot_loss(out_losses)
```

# Evaluate

```python
import denoiseg.training as tr
from tqdm.auto import tqdm
import denoiseg.evaluation as ev

model = torch.load(checkpoint)
predictions, metrics = ev.evaluate_images(
    model, 
    imgs_test,
    gts_test,
    patch_overlap = .75,
    patch_size=train_params['patch_size'],
    device = device
)

print(f'Mean IoU {np.mean(metrics):.5f}')
```

```python
import denoiseg.visualization as vis

for i,(img,gt,pred,met) in enumerate(zip(imgs_test,gts_test,predictions,metrics)):
    
    segms = []
    for im in pred[1:]:
        im = im.copy()
        im = (im - np.min(im))/(np.max(im)-np.min(im))
        im[im<.5] = 0
        im[im>=.5] = 1
        segms.append(im)

    show_imgs = [img,gt,*segms,*pred]
    vis.plot_row(show_imgs,vmin_vmax = (0,1),figsize=(50,10))
    plt.suptitle(f'Metric: {met}',y=0.72)
    plt.tight_layout()
    plt.show()
```

```python
from tqdm.auto import tqdm
import sys
sys.path.insert(0,'..')

import json
import matplotlib.pyplot as plt

import precipitate_evaluation as pv



# def save_evaluations(
#     eval_root,
#     evaluations,
#     loss_dict = None,
#     ax_figsize = 10
# ):  
#     if loss_dict is not None:
#         fig = _plot_loss(loss_dict)
#         fig.savefig(eval_root/"loss_figure.png")
#         plt.close()
    
#     for k,v in evaluations.items():
#         eval_path = eval_root/k
#         eval_path.mkdir(parents = True,exist_ok = True)
#         img_dict = v['images']
        
#         thresholds = [ vv['threshold'] for vv in v['samples']]
        
#         imgs = img_dict | {
#             f"pred_{thr:.2}":np.uint8(img_dict['foreground']>thr) 
#             for thr in thresholds
#         }
        
#         # save img
#         vis._save_imgs(eval_path, imgs)
                
#         # save fig
#         fig = vis.plot_evaluation(imgs,ax_figsize)
#         fig.suptitle(k)
#         fig.savefig(eval_path/f"{k}_plot.png")
#         plt.close()
        
#         # json
#         json.dump(v['samples'], open(eval_path/'samples.json','w'), cls=NpEncoder)

# class NpEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         if isinstance(obj, np.floating):
#             return float(obj)
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return super(NpEncoder, self).default(obj)

        
# def evaluate_and_save(model,eval_root,test_targets,crop_size):
#     eval_root.mkdir(exist_ok=True,parents=True)
    
#     evaluations = evaluate_model(model,test_targets,crop_size)
    
#     mean_evaluations = _mean_evaluations(evaluations)
    
#     best_res = _extract_best_results(mean_evaluations)
#     save_evaluations(eval_root,evaluations ,loss_dict = None)
    
#     plot_precision_recall_curve(mean_evaluations)

#     plt.savefig(eval_root/'all_prec_rec_curve.png')
    
#     json.dump(best_res, open(eval_root/'best_results.json','w'))
#     return evaluations

evaluations = pv.evaluate_match(imgs_test,gts_test,predictions)

```

```python
mean_evaluations = pv.mean_evaluations(evaluations)
    
best_res = pv.extract_best_results(mean_evaluations)
#save_evaluations(eval_root,evaluations ,loss_dict = None)
vis.plot_precision_recall_curve(mean_evaluations)
fig_path = checkpoint.parent/'fig_prec_rec.png'
plt.savefig(fig_path)
print('saved to ',fig_path)
```

```python
import pathlib
import imageio
import sys
import precipitates.dataset
import matplotlib.pyplot as plt
import itertools
from precipitates.img_tools import img2crops
import precipitates.precipitate

import matplotlib.pyplot as plt
import imageio

import precipitates.img_tools as it
import numpy as n
import json

import numpy as np
from tqdm.auto import tqdm
import cv2
import precipitates.nnet as nnet
import scipy.optimize

import pandas as pd

import logging

logger = logging.getLogger("pred")

def calculate_metrics(pred,label,clusters = [0,20,50,100,500,1024**2],component_limit = 500):
    df,p_precs,l_precs =  match_precipitates(pred,label,component_limit = component_limit)
    
    df['pred_area_px'] = [np.sum(p_precs==pred_id) for pred_id in df.pred_id]
    df['label_area_px'] = [np.sum(l_precs==label_id) for label_id in df.label_id]
    df['size_category'] = [ _category(row,clusters) for row in df.itertuples()]
    
    cat_size = {u:df[df.size_category == u] for u in df.size_category.unique()}
    cat_size[-1] = df
    
    metrics = {}
    for u,sc_df in cat_size.items():
        p,r = _prec_rec(sc_df)
        metrics[int(u)] = {
            "precision":float(p),
            "recall":float(r),
            "f1":float(_f1(p,r))
        }
    
    return metrics

```

```python
import denoiseg.instance_analysis as ia

for image,mask_f in zip(imgs_lbl, gts_lbl):
    mask = np.uint8(mask_f>0)
    df  = ia.extract_instance_properties_df(image,mask)
    
    break
df
```

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt


import imageio
import json


vis.plot_histograms(df)
plt.show()
vis.plot_instance_details(df)
plt.show()

```

```python

```
