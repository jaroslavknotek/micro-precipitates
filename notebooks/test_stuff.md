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
    logger  =_setup_logger('pred',path = '../rev-results.log')
```

```python
import sys
sys.path.insert(0,'..')
import precipitates.nnet as nnet
```

```python
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

```

```python
import precipitates.dataset as ds
import pathlib
data_20230623_root = pathlib.Path('../data/20230623/labeled/')
data_20230823_root = pathlib.Path('../data/20230823_rev/labeled/')

data_root = data_20230823_root
data_denoise_root = pathlib.Path('../../delisa-all-data/')
```

```python
train_params = {
    'train_denoise_weight':1,
    'val_denoise_weight':0,
    'patience':20,
    'segmentation_dataset_path':data_root,
    'denoise_dataset_path':data_denoise_root,
    "crop_size":128,
    "note":"fair distribution of ids"
    }
```

```python
data_test_root = pathlib.Path('../data/test/')

result_root = pathlib.Path('../rev-results')

named_data_test= ds.load_img_mask_pair(data_test_root,append_names=True)

dataset_segm = ds.load_img_mask_pair(train_params['segmentation_dataset_path'])

denoise_paths = ds._filter_not_used_denoise_paths(train_params['segmentation_dataset_path'],train_params['denoise_dataset_path'])

denoised_imgs = [ds.load_image(d) for d in denoise_paths]
data_denoised = list(zip(denoised_imgs,[None]*len(denoised_imgs)))

dataset = dataset_segm
f"{len(dataset)=},{len(named_data_test[0])=},{len(data_denoised)=}"

```

```python
from tqdm.auto import tqdm

from functools import partial
import itertools

import numpy as np
import torch
```

# Training

```python
def _train_epoch(
    model,
    dataloader,
    optimizer,
    calc_loss_fn,
    denoise_loss_weight,
    device='cpu'
):
    train_losses = []

    for targets in dataloader:
        optimizer.zero_grad()

        ls_seg,ls_denoise = _predict(model,calc_loss_fn,targets)
        
        ls = ls_seg + ls_denoise*denoise_loss_weight
        ls.backward()
        optimizer.step()
        train_losses.append(ls.item())
    
    return np.mean(train_losses)

def _predict(model,calc_loss_fn,targets):
    gpu_targets = {k:v.to(device) for k,v in targets.items()}
    pred = model(gpu_targets['x'])
    return calc_loss_fn(pred,gpu_targets)
    
def _evaluate_epoch(
    model,
    dataloader,
    calc_loss_fn,
    denoise_loss_weight,
    device = 'cpu'
):
    model.eval()
    val_losses = []
    with torch.no_grad():
        for targets in dataloader:
            ls_seg,ls_denoise = _predict(model,calc_loss_fn,targets)
            ls = ls_seg + ls_denoise*denoise_loss_weight
            val_losses.append(ls.item())
    return np.mean(val_losses)

def _get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def train_model(
    model,
    train_dataloader,
    val_dataloader,
    calc_loss_fn,
    checkpoint_path = None,
    train_loss_denoise_weight = 1,
    val_loss_denoise_weight = 1,
    device = 'cpu',
    patience = np.inf,
    epochs = 100
):

    checkpointer = MetricCheckpointer(model,checkpoint_path)
    early_stopper = nnet.EarlyStopper(patience=patience, min_delta=0)
    epochs_l = range(epochs) if epochs is not None else itertools.count()
    
    train_losses = []
    validation_losses = []
    
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(),lr = .001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        'min',
        patience = patience //2
    )
    
    for epoch in tqdm(epochs_l,desc='Training epochs'):        
        train_loss = _train_epoch(
            model,
            train_dataloader,
            optimizer,
            calc_loss_fn,
            train_loss_denoise_weight,
            device = device
        )
        
        validation_loss = _evaluate_epoch(
            model,
            val_dataloader,
            calc_loss_fn,
            val_loss_denoise_weight,
            device = device
        )
        
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        scheduler.step(validation_loss)
        
        learning_rate = _get_lr(optimizer)
        logger.info(
            f"{epoch=} {validation_loss=:.5f} {early_stopper.counter=} {learning_rate=}"
        )
        
        
        before_val_loss = checkpointer.min_validation_loss
        if checkpointer.checkpoint_if_best(validation_loss):
            best_val_loss = checkpointer.min_validation_loss
            logger.info(f"checkpoint best model {before_val_loss=:.5} -> {best_val_loss:.5}")
            
        if early_stopper.early_stop(validation_loss):             
            logger.info(f"early stopping val:{early_stopper.min_validation_loss:.5}")
            break
        
    loss_dict = {
        "train_loss":train_losses,
        "val_loss":validation_losses
    }
    
    return model, loss_dict
```

```python
from types import SimpleNamespace

import precipitates.dataset as ds
import precipitates.training as training
        
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


def get_loss(loss_name,device = 'cpu'):
    seg_loss = nnet.resolve_loss(loss_name).to(device)
    loss_denoise = torch.nn.MSELoss().to(device)
    
    def calc_loss(prediction, targets):
        
        y_segmentation = targets['y_segmentation']
        weight_map = targets['weight_map']
        has_label = targets['has_label']
        
        pred_segm = prediction[:,1:]
        
        ls_seg_pure = seg_loss(pred_segm,y_segmentation)
        ls_seg_only_valid = ls_seg_pure*weight_map*has_label
        ls_seg = ls_seg_only_valid.mean()
        
        pred_denoise = prediction[:,0][:,None,...]
        y_denoise = targets['y_denoise']
        mask_denoise = targets['mask_denoise']
        pred_denoise_masked = pred_denoise *mask_denoise
        y_denoise_masked = y_denoise * mask_denoise
        ls_denoise = loss_denoise(pred_denoise_masked, y_denoise_masked)

        return ls_seg, ls_denoise

    return calc_loss

def _train_run(
    model_eval_root,
    args,
    dataset_array,
    device, 
    patience,
    repeat
):
    apply_weight_map = args.apply_weight_map == 1
    
    # exclude only_denoise images
    if args.loss_denoise_weight == 0:
        dataset_array = [ (img,mask) for img,mask in dataset_array if mask is not None]
    
    train_dataloader,val_dataloader = ds.prepare_train_val_dataset(
        dataset_array,
        args.crop_size,
        apply_weight_map,
        repeat = repeat,
        val_size = .2
    )
    model = nnet.UNet(
        start_filters=args.cnn_filters, 
        depth=args.cnn_depth, 
        in_channels=3,
        out_channels=4
        #up_mode='bicubic'
    )
    
    train_loss_denoise_weight = args_dict['train_loss_denoise_weight']
    val_loss_denoise_weight = args_dict['val_loss_denoise_weight']
    
    loss = get_loss('fl',device = device)
    best_model_path = model_eval_root/'model-best.torch'
    model,loss_dict = train_model(
        model,
        train_dataloader,
        val_dataloader,
        train_loss_denoise_weight = train_loss_denoise_weight,
        val_loss_denoise_weight = val_loss_denoise_weight,
        checkpoint_path = best_model_path,
        calc_loss_fn = loss,
        device = device,
        epochs = 200,
        patience = patience
    )

    return model,loss_dict

class MetricCheckpointer:
    def __init__(self, model, model_path , min_delta=0):
        self.model = model
        self.model_path = model_path
        self.min_validation_loss = np.inf
        self.min_delta = min_delta

    def checkpoint_if_best(self, validation_loss):
        
        if self.model_path is None:
            return False
        
        if validation_loss > self.min_validation_loss + self.min_delta:
            return False
    
        self.min_validation_loss = validation_loss
        torch.save(self. model,self.model_path)
        return True

def run_w_config(
    args_dict,
    dataset_array, 
    named_test_data,
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
    
    try:    
        device = torch.device(device_name)
        
        model_eval_root.mkdir(exist_ok=True,parents=True)
        model,loss_dict = _train_run(model_eval_root,args,dataset_array,device,patience,repeat)    
        model_tmp_path =model_eval_root/ f'model-last.torch'
        
        torch.save(model, model_tmp_path)
        logger.info('Saved last model')
        
        
        # model.eval()
        # test_data,test_names = named_test_data
        
#         with torch.no_grad():
#             evaluations = evaluation.evaluate_model(model, test_data,test_names,args.crop_size)
            
#         visualization.save_evaluations(model_eval_root, evaluations,loss_dict)
        
        
#         aggregated = {}
#         metrics_list = [v['metrics'][-1] for k,v in evaluations.items()]
#         for k in metrics_list[0]:
#             aggregated[k] = np.mean([d[k] for d in metrics_list])    
#             aggregated[k] = np.mean([d[k] for d in metrics_list])
  
#         logger.info('Saved evaluation visualization')
        return model_eval_root,loss_dict
    except Exception as e:
        logging.error("e", exc_info=True)
```

```python
import pandas as pd

args_dict={
    'apply_weight_map':1,
    'crop_size':train_params['crop_size'],
    'cnn_depth':5,
    'loss':'fl',
    'train_loss_denoise_weight':train_params['train_denoise_weight'],
    'val_loss_denoise_weight':train_params['val_denoise_weight'],
    'cnn_filters':8
}

```

```python

final_dataset = dataset + data_denoised
if args_dict['loss_denoise_weight'] != 0:
    final_dataset = dataset + data_denoised
    
logger.info(f"len: {len(final_dataset)=}")
eval_root,loss_dict = run_w_config(
    args_list[0],
    final_dataset,
    named_data_test,
    result_root,
    patience=train_params['patience'],
    repeat=50
) 
```

# Evaluate

```python

data_test = pathlib.Path('../data/test/')
test_data_pairs, test_names = ds.load_img_mask_pair(data_test, append_names=True)


import precipitates.evaluation as ev
model_eval_root = pathlib.Path('../results-tmp/')

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
    test_data_pairs,
    test_data_names,
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
    for (test_x,test_y),name in zip(test_data_pairs,test_data_names):
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
        
        evaluations.setdefault(name.stem,{})['samples'] = imgs_prec_rec
        evaluations.setdefault(name.stem,{})['images'] = img_dict
        
    return evaluations


eval_root.mkdir(exist_ok=True,parents=True)
model = torch.load(eval_root/'model-best.torch')
evaluations = evaluate_model(model,test_data_pairs,test_names,crop_size)

```

```python
import json
import precipitates.visualization as vis

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
        # FIX
        
        
        imgs_prec_rec = v['samples']
        
        thresholds = [ vv['threshold'] for vv in imgs_prec_rec]
        precisions = [ vv['precision'] for vv in imgs_prec_rec]
        recalls = [ vv['recall'] for vv in imgs_prec_rec]
        f1s = [ vv['f1'] for vv in imgs_prec_rec]
        
        imgs = img_dict | {
            f"pred_{thr:.2}":np.uint8(img_dict['foreground']>thr) 
            for thr in thresholds
        }
        
        # save img
        vis._save_imgs(eval_path, imgs)
        
        fig,ax = plt.subplots(1,1)
        plot_precision_recall_curve(precisions,recalls,f1s,thresholds,ax=ax)
        plt.suptitle(f"{eval_root.stem}\n{k}")
        plt.tight_layout()
        fig.savefig(eval_path/f"prec_rec_plot.png")
        plt.close()
        
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
```

```python
import json
import matplotlib.pyplot as plt


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
    

mp =np.nanmean(precss,axis=0) 
rc = np.nanmean(recallss,axis=0)
mf1 = [f1(p,r) for p,r in zip(mp,rc)] 
mthrs = np.mean(thrss,axis=0)


best_id = np.argmax(mf1)

best_res = {
    "f1":mf1[best_id],
    "threshold" : mthrs[best_id],
    "precision" : mp[best_id],
    "recall": rc[best_id],
}

res_logger  =_setup_logger('results',path = f'../training_arguments_with_results.log')
res_logger.info(f"{ {'args':args_dict, 'best':best_res} }")


def _plot_f1_background(ax):
    nn=100
    x = np.linspace(0, 1, nn)
    y = np.linspace(0, 1, nn)
    xv, yv = np.meshgrid(x, y)

    f1_nn = np.array([ f1(yy,xx) for yy in y for xx in x ])
    f1_grid = (f1_nn.reshape((nn,nn)) % .1) > .05
    ax.imshow(f1_grid,alpha = .1,cmap='gray', extent=[0,1,1,0])

def plot_precision_recall_curve(precs,recs,f1s,thresholds,ax = None):
    if ax is None:
        _,ax = plt.subplots(1,1)
    
    _plot_f1_background(ax)
    
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
save_evaluations(eval_root,evaluations ,loss_dict = None)
```

```python
plot_precision_recall_curve(
    mp,
    rc,
    mf1,
    mthrs
)

plt.savefig(eval_root/'all_prec_rec_curve.png')
```

```python

```
