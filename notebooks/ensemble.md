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
```

```python
import os
#os.environ['CUDA_VISIBLE_DEVICES']=""
import tensorflow as tf
```

```python
import pathlib

model_paths = list(pathlib.Path("../../tmp").rglob("*.h5"))

assert len(model_paths)>0
print(len(model_paths))
```

```python
# relu depth=3 filters = 8 crop_size = 128 filter_size = 0 loss = bc

def _is_picked_model(name):    
    # all_conds =[
    #     #["cnn_activation=relu","cnn_depth=4","cnn_filters=8","crop_size=128","filter_size=0","loss=bfl"],
    #     ["cnn_activation=elu","cnn_depth=4","cnn_filters=8","crop_size=64","filter_size=0","loss=bfl"],
    #     ["cnn_activation=elu","cnn_depth=3","cnn_filters=16","crop_size=64","filter_size=0","loss=bfl"],
    #     ["cnn_activation=elu","cnn_depth=3","cnn_filters=16","crop_size=64","filter_size=0","loss=bc"],
    # ]
        
    all_conds = [
    ['loss=bfl','filter_size=0','cnn_filters=8','cnn_depth=5','cnn_activation=elu','crop_size=256'],
    ['loss=bfl','filter_size=0','cnn_filters=8','cnn_depth=6','cnn_activation=elu','crop_size=256'],
    ['loss=bfl','filter_size=0','cnn_filters=8','cnn_depth=7','cnn_activation=elu','crop_size=512']
    ]
    for conditions in all_conds:
        if all([c in name for c in conditions]):
            return True
    return False


picked_models_paths = [p for p in  model_paths if _is_picked_model(p.stem)]
print(len(picked_models_paths))
for p in picked_models_paths:
    print(p)
```

```python
import sys
sys.path.insert(0,'..')
```

```python
import precipitates.nn as nn
from tqdm.auto import tqdm
import re

def _load_model(model_path):
    crop_size=int(re.search('crop_size=([0-9]+)',model_path.stem).groups()[0])
    depth=int(re.search('cnn_depth=([0-9])',model_path.stem).groups()[0])
    activation=re.search('cnn_activation=([a-z]+)',model_path.stem).groups()[0]
    filters = int(re.search('cnn_filters=([0-9]+)',model_path.stem).groups()[0])
    model = nn.build_unet(
        crop_size = crop_size,
        depth= depth,
        start_filters=filters,
        activation = activation
    )
    model.load_weights(model_path)
    return model
    
#     return model

models_loaded =[_load_model(model_path) for model_path in tqdm(picked_models_paths,desc='loading models')]
```

```python
import numpy as np

class EnsembleWrapper:
    def __init__(self,models):
        self.models = models
        # sizes=set([ m.inputs[0].shape[1] for m in models])
        # assert len(sizes) ==1, "Only one size of input shape allowed"
        # self.inputs = [np.zeros((1,next(iter(sizes)),1,1))]
        
    def predict(self,*args,**kwargs):
        predictions = np.stack(
            [model.predict(*args,**kwargs) for model in self.models]
        )

        pred = np.mean(predictions, axis=0)
        return np.expand_dims(pred,axis=3)
    
    def predict_ens(self,img):
        predictions = np.stack(
            [nn.predict(model,img) for model in self.models]
        )

        pred = np.mean(predictions, axis=0)
        return pred
        return np.expand_dims(pred,axis=3)
        
        

ensemble_model = EnsembleWrapper(models_loaded)

# models = list(models_loaded)
# models.append(ensemble_model)
# model_names = [ p.stem for p in picked_models_paths]
# model_names.append("ensemble")

#mean_ens_res = evaluation.evaluate(ensemble_model,img/255,ground_truth,0)
```

```python
import precipitates.evaluation as evaluation
        
# test_dir = pathlib.Path("../data/test/PH")
# test_img_mask_pairs = evaluation._read_test_imgs_mask_pairs(test_dir)

# model_eval_PH = []
# for model in  tqdm(models,desc='eval'):
#     model_eval_PH.append([_test_model_img(model,img,mask) for i,(img,mask) in enumerate(test_img_mask_pairs)])
        
        
# test_dir = pathlib.Path("../data/test/JR")
# test_img_mask_pairs = evaluation._read_test_imgs_mask_pairs(test_dir)

# model_eval_JR = []
# for model in  tqdm(models,desc='eval'):
#     model_eval_JR.append([_test_model_img(model,img,mask) for i,(img,mask) in enumerate(test_img_mask_pairs)])
    
test_dir = pathlib.Path("../data/test/IN")
test_img_mask_pairs = evaluation._read_test_imgs_mask_pairs(test_dir)

model_eval_IN = []
for model in  tqdm(models,desc='eval'):
    model_eval_IN.append(
        [
            evaluation.evaluate(model,img,mask,0) 
            for i,(img,mask) in enumerate(test_img_mask_pairs)
        ]
    )
```

```python
import precipitates.dataset as ds
import imageio
prelabels_root = pathlib.Path("../../20230717-delta/")
prelabels_orig = np.array(list(prelabels_root.rglob("*.tif")))
np.random.seed = 123
np.random.shuffle(prelabels_orig)
top_twenty_paths = prelabels_orig[:20]

top_twenty = [ds.load_image(p) for p in top_twenty_paths]
top_twenty_paths

```

```python
import matplotlib.pyplot as plt
img = top_twenty[0]
preds = [  ensemble_model.predict_ens(img) for img in tqdm(top_twenty)]
```

```python
plt.imshow(img)    
```

```python
for p,img,pred in zip(top_twenty_paths,top_twenty, preds):
    _,axs = plt.subplots(1,2,figsize=(20,10))
    axs[0].imshow(img)
    axs[1].imshow(pred)
    plt.suptitle(p)
    plt.show()
    
```

```python
thr = .8

for p,img,pred in zip(top_twenty_paths,top_twenty, preds):
    p_str = str(p)
    p_init_str = p_str.replace('.tif','_init.png')
    p_init = pathlib.Path(p_init_str)
    
    imageio.imwrite(p_init,np.uint8(pred>np.max(pred)*thr)*255)    
```

```python
import matplotlib.pyplot as plt
import precipitates.visualization as visualization
def visu_pair(res):
    _,axs = plt.subplots(1,2,figsize = (18,9))
    for ax,(img,ground_truth,pred,metrics_res) in zip(axs,res):
        contours = visualization.add_contours_morph(img,ground_truth,contour_width=3)
        contours = visualization.add_contours_morph(contours,pred,color_rgb=(0,255,0),contour_width=2)
        m = metrics_res[-1]
        title = f"precision: {m['precision']:.3f} recall: {m['recall']:.3f} f1: {m['f1']:.3f}"
        ax.set_title(title)
        ax.imshow(contours)
        ax.plot([],c='#00FF00',label='prediction')
        ax.plot([],c='#FF0000',label='ground truth')
        ax.legend()
        ax.axis('off')
        
for x,title in zip(model_eval_IN,model_names):
    visu_pair(x)
    plt.suptitle(title)
    plt.show()
    
```
