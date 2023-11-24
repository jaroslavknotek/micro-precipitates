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
    
try:
    print(res_logger)
except NameError:
    res_logger  =_setup_logger('results',path = f'../training_arguments_with_results.log')

```

```python
import sys
import pathlib
sys.path.insert(0,'..')
import precipitates.nnet as nnet
```

```python
import torch
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

```

```python
import precipitates.dataset as ds
import pathlib
data_20230623_root = pathlib.Path('../data/20230623/labeled/')
data_20230911_root = pathlib.Path('../data/20230911_rev/labeled/')
data_20230921_root = pathlib.Path('../data/20230921_rev/labeled/')

data_root = pathlib.Path( '../../spacer_grids/labeled')
data_denoise_root = pathlib.Path( '../../spacer_grids/not_labeled')
data_test_root = pathlib.Path( '../../spacer_grids/test')

result_root = pathlib.Path('../rev-results')
model_eval_root = pathlib.Path('../results-tmp/')
```

```python
train_params = {
    'train_denoise_weight':.01,
    'val_denoise_weight':.01,
    'unet_weight_map_separation_weight':0,
    'unet_weight_map_artifacts_weight':0,
    'patience':20,
    'repeat': 50,
    'segmentation_dataset_path':data_root,
    'denoise_dataset_path':data_denoise_root,
    "crop_size":128,
    "val_size":.4,
    "note":"denoise weight + add noise",
    "augumentation_gauss_noise_val" :.005,
    "augumentation_preserve_orientation":True
}
```

```python

segmentation_targets = list(ds.get_img_dict_targets(train_params['segmentation_dataset_path']))
test_targets = list(ds.get_img_dict_targets(data_test_root))

# denoise_paths = ds._filter_not_used_denoise_paths(
#     train_params['segmentation_dataset_path'],
#     train_params['denoise_dataset_path']
# )
denoise_paths = data_denoise_root.glob('*.png')

denoised_imgs = [ds.load_image(d) for d in denoise_paths]
data_denoised = list(zip(denoised_imgs,[None]*len(denoised_imgs)))

f"{len(segmentation_targets)=},{len(test_targets)=},{len(data_denoised)=}"

```

```python
from tqdm.auto import tqdm

from functools import partial
import itertools

import numpy as np
import torch
```

```python
def fair_split_train_val_indices_to_batches(labels,batch_size,val_size):
    is_denoise = np.array([ l is None for l in labels ])

    batches = batch_fair(
        is_denoise,
        batch_size
    )

    assert np.unique([ len(b) for b in batches]) == [batch_size]
    assert len(np.unique(np.array(batches).flatten())) == len(labels)
    assert batches.shape[0] * batches.shape[1] >= len(labels)

    val_batches_num = int(np.ceil(len(batches)* val_size))
    val_batches = batches[-val_batches_num:]
    train_batches = batches[:-val_batches_num]

    train_idx = np.concatenate(train_batches)
    val_idx = np.concatenate(val_batches)

    return train_idx,val_idx

def batch_fair(is_denoise,batch_size):
    denoise_idx = np.argwhere(is_denoise).flatten().copy()
    segmantation_idx = np.argwhere(~is_denoise).flatten().copy()
    np.random.shuffle(denoise_idx)
    np.random.shuffle(segmantation_idx)

    denoised = _resize_to_shape_fill_with_random(denoise_idx,batch_size)
    segmentation = _resize_to_shape_fill_with_random(segmantation_idx,batch_size)

    return np.vstack([denoised,segmentation]).astype(int)

def _resize_to_shape_fill_with_random(arr,shape_y):
    n = len(arr)
    x_divisible = int(np.ceil(n/shape_y)) *shape_y
    to_add = x_divisible - n

    rest_idx = np.linspace(0,n-1,to_add).astype(int)
    rest = [arr[idx] for idx in rest_idx]
    flat = np.concatenate([arr,rest])
    return flat.reshape((-1,shape_y))

n = 1711
is_denoise_test = np.random.uniform(size = (n))
is_denoise_test = is_denoise_test> .7
batch_fair(is_denoise_test,32).shape
```

```python
import albumentations as A
import cv2

def get_train_val_augumentation(
    crop_size,
    preserve_orientation = False,
    noise_val = 0,
    interpolation=cv2.INTER_CUBIC
):
    crop_size_padded = int(crop_size*1.5)
    transform_list = [
        A.PadIfNeeded(crop_size_padded,crop_size_padded),
        A.RandomCrop(crop_size_padded,crop_size_padded),
        A.ElasticTransform(
                p=.5,
                alpha=10, 
                sigma=120 * 0.1,
                alpha_affine=120 * 0.1,
                interpolation=interpolation
            ),
        A.RandomBrightnessContrast(p=0.5)
    ]
    if noise_val > 0:
        transform_list.append(
            A.augmentations.transforms.GaussNoise(noise_val,p = 1), 
        )
    if not preserve_orientation:
        transform_list += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=90, interpolation=interpolation),
        ]
    else:
        transform_list.append(A.Rotate(limit=10, interpolation=interpolation))
    
    transform_list.append(A.CenterCrop(crop_size,crop_size))
    train_transform = A.Compose(transform_list)
    
    val_transform = A.Compose([
        A.PadIfNeeded(crop_size,crop_size),
        A.RandomCrop(crop_size,crop_size),
    ])
    
    return train_transform,val_transform

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        images,
        labels,
        crop_size,
        transform,
        artifact_weight_maps=None, 
        repeat = 1
    ):
        self.images = images
        self.labels = labels
        self.artifact_weight_maps = artifact_weight_maps
        
        self.transform = transform
        self.crop_size = crop_size
        self.repeat = repeat
        
    def __len__(self):
        return len(self.images) * self.repeat
    
    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError()
        
        idx = idx % len(self.images)
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.artifact_weight_maps is not None:
            art_weight_map = self.artifact_weight_maps[idx]
        else :
            art_weight_map = None
            
        if art_weight_map is None:
            art_weight_map = np.ones_like(image)
            

        has_label = label is not None
        if not has_label:
            label = np.zeros_like(image)

    
        masks = np.stack([label,art_weight_map])
        transformed = self.transform(image=image ,masks=masks)

        image = transformed['image']

        noise_y = np.copy(image)[None,...]
        noise_x = np.copy(noise_y)

        mask, mask_ind,replacement_ind = ds._get_mask(self.crop_size)

        noise_x[:, mask_ind[0],mask_ind[1]] = noise_x[:,replacement_ind[0],replacement_ind[1]]

        foreground,art_weight_map = transformed['masks']
        background = np.abs(foreground -1)
        border = ds._get_border(foreground)

        art_weight = train_params['unet_weight_map_artifacts_weight']
        sep_weight = train_params['unet_weight_map_separation_weight']
        
        if sep_weight == 0:
            sep_weight_map = np.ones(art_weight_map.shape)
        else:
            wc = {
                0: 1, # background
                1: sep_weight +1  # objects
            }
            sep_weight_map = ds.unet_weight_map(foreground, wc)
        
        weight_map = np.float32((art_weight_map -1) * art_weight + (sep_weight_map - 1) + 1)
        
        y = np.stack([foreground,background,border])
        x =  np.concatenate([noise_x]*3,axis=0)
        has_label = np.expand_dims(np.array([has_label]),axis=(1,2,3))

        return {
            'x':x,
            'y_denoise':noise_y, 
            'mask_denoise':mask[None,...],
            'y_segmentation':y,
            'weight_map':weight_map[None,...],
            'has_label':has_label
        }
    
def index_list_by_list(_list,indices):
    return [_list[i] for i in indices]


def prepare_train_val_dataset(
    images,
    labels,
    crop_size,
    artifact_weight_maps = None,
    val_size = .2,
    batch_size = 32,
    repeat = 1
):
    
    train_t,val_t = get_train_val_augumentation(
        crop_size,
        noise_val = train_params['augumentation_gauss_noise_val'],
        preserve_orientation=train_params['augumentation_preserve_orientation']
    )
    
    if artifact_weight_maps is None:
        artifact_weight_maps = [None]*len(images)
    
    train_idc, val_idc = fair_split_train_val_indices_to_batches(
        labels,
        batch_size,
        val_size
    )
    
    total_dataset_len = len(images)
    val_count = int(total_dataset_len * val_size)
    train_count = total_dataset_len -  val_count 
    
    train_dataset = Dataset(
        index_list_by_list(images,train_idc) ,
        index_list_by_list(labels,train_idc),
        crop_size,
        train_t,
        artifact_weight_maps = index_list_by_list(artifact_weight_maps,train_idc),
        repeat=repeat
    )
    # Don't shufle when using fair split
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    val_dataset = Dataset(
        index_list_by_list(images,val_idc),
        index_list_by_list(labels,val_idc),
        crop_size,
        val_t,
        artifact_weight_maps = index_list_by_list(artifact_weight_maps,val_idc),
        repeat=repeat
    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader,val_dataloader
```

```python
import matplotlib.pyplot as plt
def plot_img_row(imgs,img_size = 4 ):
    n = len(imgs)
    fig,axs  = plt.subplots(1,n,figsize = ( img_size*n,img_size))
    
    for ax,img in zip(axs,imgs):
        ax.imshow(img)    
    
    return fig,axs
```

```python
import itertools
def to_mask(a):
    if a is None:
        return None
    return np.uint8(a>0)

def to_weight_map(a):
    if a is None:
        return None
    
    return a + 1

def _sample_ds(targets,crop_size,use_weightmaps = True):
    images = []
    masks = []
    artifact_weight_maps = []
    for t in targets:
        images.append(t['img'])
        masks.append(to_mask(t.get('mask',None)))
        artifact_weight_maps.append(to_weight_map(t.get('weightmap',None))) 
        
    train_t,val_t = get_train_val_augumentation(
        crop_size,
        noise_val = train_params['augumentation_gauss_noise_val'],
        preserve_orientation=train_params['augumentation_preserve_orientation']
    )

    train_dataset = Dataset(
        images,
        masks,
        crop_size,
        train_t,
        artifact_weight_maps= artifact_weight_maps,
        repeat=1
    )
    
    tts = ( t for t in train_dataset if np.sum(t['has_label'])>0 )
    items = []
    for t in itertools.islice( tts,0,5):
        img = t['x'][0]
        mask = t['y_segmentation'][0]
        borders = t['y_segmentation'][2]
        wm = np.squeeze(t['weight_map'])       
        plot_img_row([img,mask,borders,wm])
        items.append(t)
    return items

denoise_targets = [ {'img':den}  for den in denoised_imgs]
final_targets = segmentation_targets + denoise_targets

_=_sample_ds(final_targets,128)
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
    model.train()
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
    evaluator = None,
    train_loss_denoise_weight = 1,
    val_loss_denoise_weight = 1,
    device = 'cpu',
    patience = np.inf,
    epochs = 100,
    save_epochs = 10,
):
    
    checkpointer = MetricCheckpointer(model,checkpoint_path)
    early_stopper = nnet.EarlyStopper(patience=patience, min_delta=0)
    epochs_l = range(epochs) if epochs is not None else itertools.count()
    
    train_losses = []
    validation_losses = []
    
    model.to(device)
    
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
        
        if evaluator is not None:
            evaluator.evaluate_on_epoch(model,epoch)
        
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
        artifact_weight_maps = weight_maps,
        repeat = repeat,
        val_size = train_params['val_size']
    )
    model = nnet.UNet(
        start_filters=args.cnn_filters, 
        depth=args.cnn_depth, 
        in_channels=3,
        out_channels=4
        #up_mode='bicubic'
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

def evaluate_predictions(predictions,threshold = .5,notify_progress = False):          
    evaluations = {}
    for name,record in tqdm(predictions.items(),disable = not notify_progress):
        
        foreground = record['foreground']
        y = record['y']
        t,b = split_by_mass(y)
        mask_halfs = [t,np.flip(b,axis=0)]
        t_no_stain,b_no_stains = [_perform_safe(remove_stains_half,mask_half) for mask_half in mask_halfs]
        y=np.vstack([t_no_stain,np.flip(b_no_stains,axis=0)])
        
        samples = []
        images = []
        
        clear = refine_mask_prediction(foreground,threshold = threshold)
        ld = calculate_line_distance(y,clear)
        
        evaluations.setdefault(name,{})['metrics'] = {'line_distance':ld,'threshold':threshold}
        evaluations.setdefault(name,{})['result'] = clear
        evaluations.setdefault(name,{})['prediction'] = record
        
    return evaluations

      
def _collect_predictions(model,targets,crop_size,device='cuda',notify_progress = False):
    predictions = {}
    for target in tqdm(test_targets,disable = not notify_progress):
        name = target['filename']
        test_x = target['img']
        test_y = np.uint8(target['mask']>0)
        img_dict = nnet.predict(model,test_x,crop_size,device=device)
        img_dict['y'] =test_y
        predictions[target['filename']] = img_dict
        
    return predictions
        
```

```python

def _try_eval():
    model = torch.load('../rev-results/20231025215432-crop_size=128-cnn_depth=5-loss=fl-train_loss_denoise_weight=0-val_loss_denoise_weight=0-cnn_filters=8/epoch_70/model.torch')
    predictions  = _collect_predictions(model,test_targets,args_dict['crop_size'],notify_progress = True)
    evaluations = evaluate_predictions(predictions,notify_progress = True)

    c = 4
    for name,evaluation in evaluations.items():
        fig,ax = plt.subplots(1,1,figsize = (c,c))
        image = evaluation['result']
        ax.imshow(image)
        metrics = evaluation['metrics']
        thr = metrics['threshold']
        line_distance = metrics['line_distance']
        ax.set_title(f"{thr:.2f} - {line_distance}")
        plt.show()
    return predictions,evaluations
        
predictions,evaluations = _try_eval()
```

```python

```

```python
import imageio

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
    
    distances=  np.array([ v['metrics']['line_distance'] for v in evaluations.values()])
    valid = distances[~np.isinf(distances)]
    mean_distance = np.mean(valid)
    json.dump({'line_distance':mean_distance}, open(eval_root/'avg_metrics.json','w'), cls=NpEncoder)
    
    for k,evaluation in evaluations.items():
        eval_path = eval_root/k
        eval_path.mkdir(parents = True,exist_ok = True)

        image = evaluation['result']
        imageio.imwrite(eval_path/'result.png', image*255)
        
        fig = vis.plot_evaluation(evaluation['prediction'])
        fig.savefig(eval_path/"prediction.png")
        plt.close(fig)
        # json
        
        ld = evaluation['metrics']['line_distance']
        if np.isinf(ld):
            evaluation['metrics']['line_distance'] = None
        json.dump(evaluation['metrics'], open(eval_path/'metrics.json','w'), cls=NpEncoder)
        
    
    
        

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

        
def evaluate_and_save(model,eval_root,test_targets,crop_size,notify_progress = False):
    predictions  = _collect_predictions(model,test_targets,args_dict['crop_size'],notify_progress=notify_progress)
    evaluations = evaluate_predictions(predictions,notify_progress=notify_progress)

    eval_root.mkdir(exist_ok=True,parents=True)
    
    save_evaluations(eval_root,evaluations ,loss_dict = None)  
    
    return evaluations


class EpochModelEvaluator:
    def __init__(
        self,
        targets, 
        eval_root,
        crop_size,
        evaluate_every_nth_epoch = 10,
        evaluate_after_nth_epoch = 30
    ):
        self.targets = targets
        self.eval_root = pathlib.Path(eval_root)
        self.crop_size = crop_size
        
        self.nth_epoch = evaluate_every_nth_epoch
        self.after_epoch = evaluate_after_nth_epoch
    
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
            notify_progress = True
        )

model = torch.load('../rev-results/20231025215432-crop_size=128-cnn_depth=5-loss=fl-train_loss_denoise_weight=0-val_loss_denoise_weight=0-cnn_filters=8/model-last.torch')
ech = EpochModelEvaluator(test_targets,"tmp", 128)
ech.evaluate_on_epoch(model,80)
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
