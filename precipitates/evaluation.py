import pathlib

import pathlib
import imageio
import sys
import precipitates.dataset
import matplotlib.pyplot as plt
import precipitates.nn as nn
import itertools
from precipitates.dataset import img2crops
import precipitates.precipitate

import matplotlib.pyplot as plt
import imageio

import precipitates.img_tools as it
import numpy as n
import tensorflow as tf
import json


import precipitates.nn as nn
from tqdm.auto import tqdm
import numpy as np
import pathlib
from tqdm.auto import tqdm
import cv2

import scipy.optimize

import pandas as pd

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


def _intersection(a,b):
    if a is None or b is None:
        return np.nan
    else:
        return np.sum((a*b)>0)
    
def _union(a,b):
    assert a is not None or b is not None
        
    if a is None:
        a = np.zeros(b.shape)
    if b is None:
        b = np.zeros(a.shape)
    
    return np.sum((a+b)>0)

def _category(row,clusters):
    
    vals = [x for x in [row.label_area_px,row.pred_area_px] if x is not None]
    assert len(vals) >0
    min_size = np.nanmin(vals)
    for a,b in zip(clusters,clusters[1:]):
        if a<=min_size<b:
            return clusters.index(a)
    assert False

def _merge(masks):
    masks = [mask for mask in masks if mask is not None]
    return np.uint8(np.sum(masks,axis=0)>0)
   
def _iou(label,pred):
    m = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])
    m.update_state(label,pred)
    return m.result().numpy()

def _iou_from_arr(label_arr,pred_arr):
    pred_mask = _merge(pred_arr)
    label_mask = _merge(label_arr)
    if pred_mask.shape != label_mask.shape or len(label_mask.shape) !=2:
        return np.nan
    return _iou(label_mask,pred_mask)

def _f1(precision, recall):
    if precision + recall ==0:
        return np.nan
    return 2*(precision * recall)/(precision + recall)

def _calculate_metrics(pred,label,clusters = [0,50,500,1024**2]):
    df,p_precs,l_precs =  match_precipitates(pred,label)
    
    df['pred_area_px'] = [np.sum(p_precs==pred_id) for pred_id in df.pred_id]
    df['label_area_px'] = [np.sum(l_precs==label_id) for label_id in df.label_id]
    df['size_category'] = [ _category(row,clusters) for row in df.itertuples()]
    
    cat_size = {u:df[df.size_category == u] for u in df.size_category.unique()}
    cat_size[-1] = df
    
    metrics = {}
    for u,sc_df in cat_size.items():
        # iou =_iou_from_arr(
        #     sc_df.pred_mask.to_numpy(),
        #     sc_df.label_mask.to_numpy()
        # )
        p,r = _prec_rec(sc_df)
        metrics[int(u)] = {
            #"iou":float(iou),
            "iou":-1,#disabled
            "precision":float(p),
            "recall":float(r),
            "f1":float(_f1(p,r))
        }
    
    return metrics

def _prec_rec(df):
    grains_pred = len(df[~df['pred_id'].isna()])
    grains_label = len(df[~df['label_id'].isna()])
    
    tp = df[~df['label_id'].isna() & ~df['pred_id'].isna()]
    if grains_pred !=0:
        precision = len(tp) / grains_pred
    else: 
        precision = np.nan
        
    if grains_label != 0:
        recall =  len(tp) / grains_label
    else:
        recall = np.nan
    return precision,recall

def _norm(img):
    # img_min=np.min(img)
    img_min = 0
    img_max = np.max(img)
    return (img.astype(float)-img_min) / (img_max-img_min)

def dwbce(true,pred):
    dwbce = nn.DynamicallyWeightedBinaryCrossentropy()
    return dwbce(true, pred).numpy()

def wbce(true,pred,weight_zero=1,weight_one=1):
    wbce = nn.WeightedBinaryCrossentropy(
        weight_zero= weight_zero,
        weight_one= weight_one
    )
    return wbce(true, pred).numpy()

def bce(true,pred):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return bce(true, pred).numpy()

def _zip_pred_label_crops(mask, pred,stride = 128,shape=(128,128)):
    mask_crop_sets_it = img2crops(mask.astype(np.float32),stride, shape)
    pred_crop_sets_it = img2crops(pred.astype(np.float32),stride, shape)
    return zip(mask_crop_sets_it,pred_crop_sets_it)


def evaluate(model, img, ground_truth,filter_size):
    img = _norm(img)
    
    pred = nn.predict(model,img)    
    
    # prediction is never filtered
    if filter_size > 0:  
        ground_truth = it._filter_small(ground_truth,filter_size=filter_size)
        
    metrics_res = _calculate_metrics(pred,ground_truth)
    return (img,ground_truth,pred,metrics_res)

def _read_test_imgs_mask_pairs(test_dir):
    
    img_mask_pair = []
    img_paths = pathlib.Path(test_dir).rglob("img.png")
    for img_path in img_paths:
        mask_path = img_path.parent/'mask.png'
        img = precipitates.precipitate.load_microscope_img(img_path)
        mask =imageio.imread(mask_path)
        img_mask_pair.append((img,mask))
        
    return img_mask_pair
    

def evaluate_models(
    models_paths,
    test_imgs_folder,
    filter_size =0
):
    
    test_img_mask_pairs=_read_test_imgs_mask_pairs(test_imgs_folder)
    results = []
    for model_path in tqdm(models_paths,desc = "Applying model"):
        model = nn.compose_unet((128,128))
        model.load_weights(model_path)
        
        for img,mask in test_img_mask_pairs:
            (img,ground_truth,pred,metrics_res) = evaluate(
                model,
                img,
                mask,
                filter_size
            )
            
            results.append({
                "img":img,
                "mask":ground_truth,
                "pred":pred,
                "metrics": metrics_res,
                "model_path": model_path
            })

    return results

def _visualize_pairs(axs,img,mask,pred,metrics,model_name):
    full_img_index = -1
    f1 = metrics[full_img_index]['f1']
    iou = metrics[full_img_index]['iou']
    title = f"{model_name} - f1:{f1},iou:{iou}"
    axs[1].set_title(title)
    for ax,img in zip(axs,[img,mask,pred]):
        ax.imshow(img)

    idxs = list(metrics.keys())
    ious = [v['iou'] for v in metrics.values()]
    f1s = [v['f1'] for v in metrics.values()]

    axs[-1].plot(idxs,f1s,'x',label='F1')
    axs[-1].plot(idxs,ious,'x',label='IOU')
    axs[-1].legend()

def _visualize(results):
    fig,axs = plt.subplots(len(results),4,figsize=(16,4*len(results)))
    for ax_r,row in zip(axs,results):
        img = row['img']
        mask = row['mask']
        pred = row['pred']
        metrics = row['metrics']
        model_path = row['model_path']
        model_name = f"{model_path.parent.name} - {model_path.name}"
        _visualize_pairs(ax_r,img,mask,pred,metrics,model_name)
    return fig


if __name__ == "__main__":
    all_models = list(pathlib.Path('/home/jry/test-models/').rglob("*.h5"))
    
    test_data_dirs = list( pathlib.Path('data/test').rglob('*IN/img.png'))
    
    results = evaluate_models(all_models,test_data_dirs)
    fig = _visualize(results)
