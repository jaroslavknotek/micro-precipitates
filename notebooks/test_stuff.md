---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.6
  kernelspec:
    display_name: cv_torch
    language: python
    name: cv_torch
---

```python
%load_ext autoreload
%autoreload 2
```

```python
import sys
sys.path.insert(0,'..')
import precipitates.nnet as nnet
```

```python
from tqdm.auto import tqdm

from functools import partial
import itertools

import numpy as np
import torch

import logging
logging.basicConfig()
logger = logging.getLogger('pred')
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler('../x2-results.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


def _calculate_loss(
    y,
    pred,
    mask,
    has_label,
    weight_mask,
    loss_denoise,
    loss,
    denoise_loss_weight = 2):
    
    ls_segmentation = loss(pred[:,1:,...]*has_label,y[:,1:,...]*has_label)
    ls_seg_masked = (ls_segmentation*weight_mask).mean()
    
    if denoise_loss_weight == 0:
        return ls_seg_masked
    else:   
        ls_denoise = loss_denoise(pred[:,0] * mask, y[:,0] * mask)
        return ls_seg_masked + ls_denoise*denoise_loss_weight

def train_model(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    loss_denoise,
    loss,
    denoise_loss_weight = 2,
    wandb = None,
    device = 'cpu',
    patience = 5
):
    early_stopper = nnet.EarlyStopper(patience=patience, min_delta=0)
    
    loss_dict = {}
    
    epochs = itertools.count()
    for e in tqdm(epochs,desc='Training epochs'):
        model.train()
        train_losses = []
        val_losses = []

        for rec in train_dataloader:
            x, y,mask,has_label,wm = [r.to(device) for r in rec]
            optimizer.zero_grad()
            pred = model(x)
            ls = _calculate_loss(
                y,
                pred,
                mask,
                has_label,
                wm,
                loss_denoise,
                loss,
                denoise_loss_weight
            )

            ls.backward()
            optimizer.step()
            train_losses.append(ls.item())

        with torch.no_grad():
            model.eval()
            for rec in val_dataloader:
                x, y,mask,has_label,wm = [r.to(device) for r in rec]            
                pred = model(x)
                ls = _calculate_loss(
                    y,
                    pred,
                    mask,
                    has_label,
                    wm,
                    loss_denoise,
                    loss,
                    denoise_loss_weight
                )
                val_losses.append(ls.item())

        loss_dict.setdefault("train_loss",[]).append(train_losses)
        loss_dict.setdefault("val_loss",[]).append(val_losses)

        validation_loss = np.mean(val_losses)
                
        logger.info(f"epoch {e} val:{validation_loss:.5f} stopper counter:{early_stopper.counter}")
        if early_stopper.early_stop(validation_loss):             
            logger.info("early stopping")
            break
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

def _train_run(model_eval_root,args,dataset_array,device, patience,repeat):
    apply_weight_map = args.apply_weight_map == 1
    
    # exclude only_denoise images
    if args.loss_denoise_weight == 0:
        dataset_array = [ (img,mask) for img,mask in dataset_array if mask is not None]
    
    train_dataloader,val_dataloader = ds.prepare_train_val_dataset(
        dataset_array,
        args.crop_size,
        apply_weight_map,
        repeat = repeat,
        val_size = .4
    )
    model = nnet.UNet(
        start_filters=args.cnn_filters, 
        depth=args.cnn_depth, 
        in_channels=3,
        out_channels=4
        #up_mode='bicubic'
    )

    loss = nnet.resolve_loss(args.loss)
    loss_denoise = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    model.to(device)
    
        
    model,loss_dict = training.train_model(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        loss_denoise,
        loss,
        model_save_dir = model_eval_root,
        denoise_loss_weight = args.loss_denoise_weight,
        device=device,
        patience = patience,
        epochs_n = None
    )

    return model,loss_dict

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
        model.eval()
        test_data,test_names = named_test_data
        
        with torch.no_grad():
            evaluations = evaluation.evaluate_model(model, test_data,test_names,args.crop_size)
            
        visualization.save_evaluations(model_eval_root, evaluations,loss_dict)
        
        
        aggregated = {}
        metrics_list = [v['metrics'][-1] for k,v in evaluations.items()]
        for k in metrics_list[0]:
            aggregated[k] = np.mean([d[k] for d in metrics_list])    
            aggregated[k] = np.mean([d[k] for d in metrics_list])
  
        logger.info('Saved evaluation visualization')
        return aggregated
    except Exception as e:
        logging.error("e", exc_info=True)
```

```python
import precipitates.dataset as ds
data_20230623_root = pathlib.Path('../data/20230623/labeled/')

#data_20230724_root = pathlib.Path('../data/20230724/labeled/')
data_denoise_root = pathlib.Path('../../delisa-new-up/')
data_test_root = pathlib.Path('../data/test/')

result_root = pathlib.Path('../results')

named_data_test= ds.load_img_mask_pair(data_test_root,append_names=True)

data_20230623 = ds.load_img_mask_pair(data_20230623_root)

denoise_path = ds._filter_not_used_denoise_paths(data_20230623_root,data_denoise_root)
denoised_imgs = [ds.load_image(d) for d in denoise_path]
data_denoised = list(zip(denoised_imgs,[None]*len(denoised_imgs)))

#f"{len(data_20230623)=},{len(named_data_test[0])=},{len(data_denoised)=}"
```

```python
import pandas as pd

cols = [
    'apply_weight_map',
    'crop_size',
    'cnn_depth',
    'loss',
    'loss_denoise_weight',
    'cnn_filters'
]

params_data =[
    # (0,128,5,'fl',0,8),
    # (0,128,5,'fl',1,8),
    # (0,128,5,'fl',10,8),
    # (1,128,5,'fl',0,8),
    # (1,128,5,'fl',1,8),
    # (1,128,5,'fl',10,8),
    # (0,128,5,'fl',0,16),
    # (0,128,5,'fl',1,16),
    # (0,128,5,'fl',10,16),
    # (1,128,5,'fl',0,16),
    # (1,128,5,'fl',1,16),
    # (1,128,5,'fl',10,16),
    
    # (1,256,6,'fl',0,8),
    # (1,256,6,'fl',1,8),
    
    (1,256,6,'fl',10,8),
    (0,256,6,'fl',0,8),
    (0,256,6,'fl',1,8),
    (0,256,6,'fl',10,8),
    
    (1,256,6,'fl',0,16),
    (1,256,6,'fl',1,16),
    (1,256,6,'fl',10,16),
    
    (0,256,6,'fl',0,16),
    (0,256,6,'fl',1,16),
    (0,256,6,'fl',10,16)
]

df = pd.DataFrame(reversed(params_data),columns = cols)
```

```python
args_list = df.to_dict(orient='records')
SimpleNamespace(**args_list[0])
```

```python
for args_dict in args_list:
    dataset = data_20230623
    if args_dict['loss_denoise_weight'] != 0:
        dataset = dataset + data_denoised

    logger.info(f"Starging: len: {len(dataset)=}")
    logger.debug(f"config: {args_dict}")
    metrics = run_w_config(
        args_dict,
        dataset,
        named_data_test,
        result_root,
        patience=20,
        repeat=50
    ) 
    logger.info(metrics)
```
