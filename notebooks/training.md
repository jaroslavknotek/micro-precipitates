---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: napari_sam
    language: python
    name: napari_sam
---

```python
%load_ext autoreload
%autoreload 2
```

```python
import logging 
logging.basicConfig()
def _setup_logger(name,path = None,level = logging.DEBUG):
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if path is not None:
        formatter = logging.Formatter(
            '%(process)d: %(asctime)s - %(filename)s:%(funcName)s:%(lineno)d - %(levelname)s - %(message)s'
        )
        fh = logging.FileHandler(path)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
    return logger

try:
    print(logger)
except NameError:
    logger  =_setup_logger('denoiseg',path = '../rev-results.log')
    
try:
    print(res_logger)
except NameError:
    res_logger  =_setup_logger('results',path = f'../training_arguments_with_results.log')

```

```python
import sys
sys.path.insert(0,'..')
```

```python
import torch
import numpy as np
import pathlib

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

```python
import imageio
import matplotlib.pyplot as plt

root = pathlib.Path('/home/jry/data/UHCS/')
gt_root = root/'GT_bin'
img_root = root/'labeled_img'
unlabeled_root = root/'unlabeled_img'

gt_paths = list(gt_root.glob('*.png'))
img_paths = [ img_root/f"{p.stem}.jpg" for p in gt_paths]
unlabeled_paths = list(unlabeled_root.glob('*.jpg'))

```

```python
import denoiseg.image_utils as iu

imgs_unlbl = list(map(iu.load_image, unlabeled_paths))
gts_unlbl = [None]*len(imgs_unlbl)

imgs_lbl = list(map(iu.load_image, img_paths))
gts_lbl = list(map(iu.load_mask, gt_paths)) 

imgs = imgs_lbl + imgs_unlbl
gts = gts_lbl + gts_unlbl

for img,gt in zip(imgs,gts):
    _,axs = plt.subplots(1,2)
    axs[0].imshow(img)
    if gt is None:
        gt = np.zeros_like(img)
    axs[1].imshow(gt)
    plt.show()
    break

patch_size = 128
```

```python
import cv2


import itertools
import denoiseg

import denoiseg.dataset as ds

aug_train = ds.setup_augumentation(
    patch_size,
    elastic = True,
    brightness_contrast = True,
    flip_vertical = True,
    flip_horizontal = True,
    blur_sharp_power = 1,
    noise_val = .01,
    rotate_deg = 90
)
dataset = ds.DenoisegDataset(
    imgs,
    gts,
    patch_size,
    aug_train,
    repeat = 1
)
ds.sample_ds(dataset,3)
```

```python
train_params = {
    "patch_size":128,
    "validation_set_percentage":.2,
    "batch_size":32,
    "dataset_repeat":50,
    "model":{
        "filters":8,
        "depth":5,
    },
    #training
    "epochs":100,
    "patience":20,
    "scheduler_patience":10,
    "denoise_loss_weight":1,
    
    "augumentation":{
        "elastic":True,
        "brightness_contrast":True,
        "flip_vertical": True,
        "flip_horizontal": True,
        "blur_sharp_power": 1,
        "noise_val": .01,
        "rotate_deg": 90
    }
}
train_params['dataset_repeat'] = 1
train_params['epochs'] = 5
train_params['batch_size'] = 16
train_params['model']['depth'] = 3

# train_params = {
#     'train_denoise_weight':1,
#     'val_denoise_weight':1,
#     'patience':30,
#     'repeat': 50,
#     'segmentation_dataset_path':data_root,
#     'denoise_dataset_path':data_denoise_root,
#     "crop_size":128,
#     "val_size":.2,
#     "note":"repeat by 100",
#     "augumentation_gauss_noise_val" :.002,
#     "augumentation_preserve_orientation":True
# }
```

```python
train_dataloader,val_dataloader = ds.prepare_dataloaders(imgs,gts,train_params)
```

```python
# for d in train_dataloader:
#     print(d.keys())
#     for xx in [d['y_denoise'],d['y_segmentation'][:,0],d['y_segmentation'][:,1],d['y_segmentation'][:,2]]:
#         i = xx.detach().cpu().numpy()
#         print(np.min(i),np.max(i))
#         break
#     break

#     #break
```

# Training

```python
import os

```

```python
import denoiseg.unet
import denoiseg.training

os.environ['OMP_THREAD_LIMIT'] = '8'
os.environ['OMP_NUM_THREADS'] = '8'


model_config = train_params['model']
model = denoiseg.unet.UNet(
    start_filters=model_config['filters'], 
    depth=model_config['depth'], 
    in_channels=3,
    out_channels=4
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
device = None
checkpoint_path = pathlib.Path('training')/'model-best.pth'
checkpoint_path.parent.mkdir(exist_ok=True,parents=True)

loss_fn = denoiseg.training.get_loss(
    device = device,
    denoise_loss_weight = train_params['denoise_loss_weight']
)
out_losses = denoiseg.training.train(
    model,
    train_dataloader,
    val_dataloader,
    loss_fn,
    epochs = train_params['epochs'],
    patience = train_params['patience'],
    scheduler_patience = train_params['scheduler_patience'],
    checkpoint_path = checkpoint_path,
    device = device
)
    
    
```

```python
from types import SimpleNamespace

import precipitates.dataset as ds
        
import precipitates.evaluation as evaluation
import precipitates.visualization as visualization

import precipitates.nnet as nnet
from datetime import datetime
import pathlib

import argparse
import logging 

import sys
import traceback

import functools
import numpy as np

import torch
from torch import nn




def _train_run(
    model_eval_root,
    args,
    targets,
    device, 
    patience,
    repeat,
    evaluator
):    
    # exclude only_denoise images
    if args.train_loss_denoise_weight == 0:
        targets = [ t for t in targets if 'mask' in t]
    logger.info(f"len: {len(targets)=}")
    images = []
    masks = []
    weight_maps = []
    for t in targets:
        images.append(t['img'])
        masks.append(to_mask(t.get('mask',None)))
        weight_maps.append(to_weight_map(t.get('weightmap',None)))          
    
    weight_maps = [None]*len(images)
    
    train_dataloader,val_dataloader = prepare_train_val_dataset(
        images,
        masks,
        args.crop_size,
        weight_maps = weight_maps,
        repeat = repeat,
        val_size = train_params['val_size']
    )
        
    train_loss_denoise_weight = args.train_loss_denoise_weight
    val_loss_denoise_weight = args.val_loss_denoise_weight
    
    loss = get_loss('fl',device = device)
    best_model_path = model_eval_root/'model-best.torch'
    model,loss_dict = train_model(
        model,
        train_dataloader,
        val_dataloader,
        train_loss_denoise_weight = train_loss_denoise_weight,
        val_loss_denoise_weight = val_loss_denoise_weight,
        checkpoint_path = best_model_path,
        evaluator = evaluator,
        calc_loss_fn = loss,
        device = device,
        epochs = 200,
        patience = patience
    )

    return model,loss_dict

def run_w_config(
    args_dict,
    targets, 
    test_targets,
    results_dir_root,
    patience = 5,
    device_name='cuda',
    repeat = 100,
):
    ts = datetime.strftime(datetime.now(),'%Y%m%d%H%M%S')
    model_suffix = '-'.join([ f"{k}={args_dict[k]}" for k in args_dict])
    model_eval_root = results_dir_root/f"{ts}-{model_suffix}"
    model_eval_root.mkdir(parents=True,exist_ok=True)
    
    args = SimpleNamespace(**args_dict)    
    
    if np.log2(args.crop_size) < args.cnn_depth +2:
        logger.warn(f"Cannot have crop_size={args.crop_size} and cnn_depth={args.cnn_depth}")
        return
                
    evaluator = EpochModelEvaluator(
        test_targets,
        model_eval_root,
        crop_size  =args.crop_size,
        device = device_name
    )

    device = torch.device(device_name)

    model_eval_root.mkdir(exist_ok=True,parents=True)
    model,loss_dict = _train_run(
        model_eval_root,
        args,
        targets,
        device,
        patience,
        repeat,
        evaluator
    )    
    model_tmp_path =model_eval_root/ f'model-last.torch'

    torch.save(model, model_tmp_path)
    logger.info('Saved last model')

    return model_eval_root,loss_dict
```

```python
import pandas as pd

args_dict={
    'crop_size':train_params['crop_size'],
    'cnn_depth':5,
    'loss':'fl',
    'train_loss_denoise_weight':train_params['train_denoise_weight'],
    'val_loss_denoise_weight':train_params['val_denoise_weight'],
    'cnn_filters':8
}

```

# Evaluation

```python
class EpochModelEvaluator:
    def __init__(
        self,
        targets, 
        eval_root,
        crop_size,
        evaluate_every_nth_epoch = 10,
        evaluate_after_nth_epoch = 30,
        device = 'cpu'
    ):
        self.device = device
        self.targets = targets
        self.eval_root = eval_root
        self.crop_size = crop_size
        
        self.nth_epoch = evaluate_every_nth_epoch
        self.after_epoch = evaluate_after_nth_epoch

    def __call__(self, model, epoch):
        return self.evaluate_on_epoch(model,epoch)
    
    def evaluate_on_epoch(self, model, epoch):
        if epoch < self.after_epoch or epoch % self.nth_epoch != 0:
            return
        
        logger.info(f"Evaluating on {epoch=}")
        
        eval_path = self.eval_root/f'epoch_{epoch}'
        eval_path.mkdir(exist_ok=True,parents=True)
        
        model_path = eval_path /'model.torch'
        torch.save(model,model_path)
        
        evaluate_and_save(
            model,
            eval_path,
            self.targets,
            self.crop_size,
            device = self.device,
        )
        plt.close()

        
def evaluate_and_save(
    model,
    eval_root,
    test_targets,
    crop_size,
    device = 'cpu'
):
    eval_root.mkdir(exist_ok=True,parents=True)
    
    evaluations = evaluate_model(model,test_targets,crop_size,device = device)
    
    mean_evaluations = _mean_evaluations(evaluations)
    
    best_res = _extract_best_results(mean_evaluations)
    save_evaluations(eval_root,evaluations ,loss_dict = None)
    
    plot_precision_recall_curve(mean_evaluations)

    plt.savefig(eval_root/'all_prec_rec_curve.png')
    
    json.dump(best_res, open(eval_root/'best_results.json','w'))
    return evaluations


def evaluate_model(
    model,
    test_data,
    test_data_names,
    crop_size,
    segmentation_thr = .7,
    device = 'cpu',
):    
    evaluations = {}
    for (test_x,test_y),name in zip(test_data,test_data_names):
        img_dict = segmentation.segment_image(
            model,
            test_x,
            crop_size,
            device=device
        )
        
        img_dict['y'] = test_y
        evaluations.setdefault(name.stem,{})['images'] = img_dict
    
        pred = np.uint8(img_dict['foreground']>segmentation_thr)
        gt = np.uint8(img_dict['y']>0)    
        metrics_res = calculate_metrics(pred,gt)
        evaluations.setdefault(name.stem,{})['metrics'] = metrics_res
        
    return evaluations


```

```python
import json
import matplotlib.pyplot as plt
import precipitates.visualization as vis

import precipitates.evaluation as ev


def _norm(img):
    # img_min=np.min(img)
    img_min = 0
    img_max = np.max(img)
    return (img.astype(float)-img_min) / (img_max-img_min)

def _threshold_foreground(foreground,thr):
    if thr == 0:
        return np.ones_like(foreground,dtype = np.uint8)
    elif thr == 1:
        return np.zeros_like(foreground,dtype = np.uint8)
    p = np.zeros_like(foreground)
    p[foreground>=thr] = 1
    return np.uint8(p)

def f1(precision, recall):
    if precision + recall ==0:
        return np.nan
    return 2*(precision * recall)/(precision + recall)

def _calc_prec_rec_from_pred(y,p):    

    if (p == 1).all():
        return (0,1)
    elif (p == 0).all():
        return (1,0)
    
    y = np.uint8(y)
    df,_,_ = ev.match_precipitates(p,y)
    return ev._prec_rec(df)

def sample_precision_recall(ground_truth,foreground,thresholds):
    
    thr_imgs = [
        _threshold_foreground(foreground,thr) 
        for thr in thresholds
    ]
        
    precs_recs = [
        _calc_prec_rec_from_pred(ground_truth,img) 
        for img in thr_imgs
    ]
    pr = np.array(precs_recs)
    precs, recs = pr[np.argsort(pr[:,0])].T
    f1s = np.array([f1(prec,rec)  for prec,rec in zip(precs,recs)])
    
    return thr_imgs,precs,recs,f1s

def evaluate_model(
    model,
    test_targets,
    crop_size,
    device='cuda'
):  
    
    thr_low = .4
    thr_high = .8
    n = 5
    thresholds = np.concatenate([
        [0.0],
        np.linspace(thr_low,thr_high,n),
        [1.0]
    ])
    
    evaluations = {}
    for target in test_targets:
        name = target['filename']
        test_x = target['img']
        test_y = np.uint8(target['mask']>0)
        
        img_dict = nnet.predict(model,test_x,crop_size,device=device)
        img_dict['y'] = np.uint8(test_y>0)     
        
        logger.info(f"Sampling precision recall ({len(thresholds)})")
        thr_imgs,precs,recs,f1s = sample_precision_recall(
            img_dict['y'],
            img_dict['foreground'],
            thresholds
        )
        
        imgs_prec_rec = [
            {
                "precision":p,
                "recall":r,
                "f1":f,
                "threshold":t
            }
            for p,r,f,t in zip(precs,recs,f1s,thresholds)
        ]
        
        evaluations.setdefault(name,{})['samples'] = imgs_prec_rec
        evaluations.setdefault(name,{})['images'] = img_dict
        
    return evaluations


def save_evaluations(
    eval_root,
    evaluations,
    loss_dict = None,
    ax_figsize = 10
):  
    if loss_dict is not None:
        fig = _plot_loss(loss_dict)
        fig.savefig(eval_root/"loss_figure.png")
        plt.close()
    
    for k,v in evaluations.items():
        eval_path = eval_root/k
        eval_path.mkdir(parents = True,exist_ok = True)
        img_dict = v['images']
        
        thresholds = [ vv['threshold'] for vv in v['samples']]
        
        imgs = img_dict | {
            f"pred_{thr:.2}":np.uint8(img_dict['foreground']>thr) 
            for thr in thresholds
        }
        
        # save img
        vis._save_imgs(eval_path, imgs)
                
        # save fig
        fig = vis.plot_evaluation(imgs,ax_figsize)
        fig.suptitle(k)
        fig.savefig(eval_path/f"{k}_plot.png")
        plt.close()
        
        # json
        json.dump(v['samples'], open(eval_path/'samples.json','w'), cls=NpEncoder)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def _mean_evaluations(evaluations):
    thrss = []
    precss = []
    recallss = []
    f1ss = []

    for k,v in evaluations.items():
        imgs_prec_rec = v['samples']        
        thresholds = [ vv['threshold'] for vv in imgs_prec_rec]
        precisions = [ vv['precision'] for vv in imgs_prec_rec]
        recalls = [ vv['recall'] for vv in imgs_prec_rec]
        f1s = [ vv['f1'] for vv in imgs_prec_rec]

        thrss.append(thresholds)
        precss.append(precisions)
        recallss.append(recalls)
        f1ss.append(f1s)


    mean_precisions =np.nanmean(precss,axis=0) 
    mean_recalls = np.nanmean(recallss,axis=0)
    mean_f1s = [f1(p,r) for p,r in zip(mean_precisions,mean_recalls)] 
    thresholds = np.mean(thrss,axis=0)
    
    return {
        'mean_precisions':mean_precisions,
        'mean_recalls':mean_recalls,
        'mean_f1s':mean_f1s,
        'thresholds':thresholds
    }

def _extract_best_results(mean_evaluations):
    best_id = np.argmax(mean_evaluations['mean_f1s'])

    return {
        "f1":mean_evaluations['mean_f1s'][best_id],
        "threshold" : mean_evaluations['thresholds'][best_id],
        "precision" : mean_evaluations['mean_precisions'][best_id],
        "recall": mean_evaluations['mean_recalls'][best_id],
    }



def _plot_f1_background(ax,nn=100):
    x = np.linspace(0, 1, nn)
    y = np.linspace(0, 1, nn)
    xv, yv = np.meshgrid(x, y)

    f1_nn = np.array([ f1(yy,xx) for yy in y for xx in x ])
    f1_grid = (f1_nn.reshape((nn,nn)) % .1) > .05
    ax.imshow(f1_grid,alpha = .1,cmap='gray', extent=[0,1,1,0])

def plot_precision_recall_curve(mean_evaluations,ax = None):
    if ax is None:
        _,ax = plt.subplots(1,1)
    
    _plot_f1_background(ax)
    
    f1s = mean_evaluations['mean_f1s']
    thresholds =  mean_evaluations['thresholds']
    precs = mean_evaluations['mean_precisions']
    recs = mean_evaluations['mean_recalls']
    
    ax.plot(recs, precs)
    ax.set_title("Precision/Recall Chart")
    
    ax.set_xlabel('Recall')
    ax.set_xlim(0,1)
    
    ax.set_ylabel('Precision')
    ax.set_ylim(0,1)
    
    ax.axis('scaled')

    for prec,rec,f1,thr in zip(precs,recs,f1s,thresholds):
        lbl = f"$f_1$:{f1:.2} $t$:{thr:.2}"
        #ax.plot([rec],[prec],'x',label=)
        ax.text(rec,prec,lbl)

```

```python
denoise_targets = [ {'img':den}  for den in denoised_imgs]
final_targets = segmentation_targets + denoise_targets
```

```python
eval_root,loss_dict = run_w_config(
    args_dict,
    final_targets,
    test_targets,
    result_root,
    patience=train_params['patience'],
    repeat=train_params['repeat'],
    device_name = device.type
) 
```

```python
model = torch.load(eval_root/'model-best.torch')

json.dump({ k:str(v) for k,v in train_params.items()}, open(eval_root/'train_params.json','w'))

eval_path = eval_root/'best'
eval_path.mkdir(parents=True,exist_ok=True)
evaluations = evaluate_and_save(model,eval_path,test_targets,train_params['crop_size'])

mean_evaluations = _mean_evaluations(evaluations)
best_res = _extract_best_results(mean_evaluations)
res_logger.info(f"{ {'args':args_dict, 'best':best_res} }")
```

```python
exit()
```

```python

```
