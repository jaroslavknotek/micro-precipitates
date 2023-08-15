import pathlib
import sys
import itertools
from precipitates.img_tools import img2crops

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
    for p_grain_id in range(1,p_n):
        p_grain_mask = it._extract_grain_mask(p_grains,p_grain_id)

        intersecting_ids = np.unique(l_grains*p_grain_mask)
        intersecting_ids = intersecting_ids[intersecting_ids>0]
        
        for l_grain_id in intersecting_ids:
            l_grain_mask = it._extract_grain_mask(l_grains,l_grain_id)
            weight = 1 - _iou(l_grain_mask,p_grain_mask)
            weights_dict.setdefault(p_grain_id,{}).setdefault(l_grain_id,weight)
            
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
    
    
def match_precipitates(prediction,label,component_limit = 500):
    p_n, p_grains = cv2.connectedComponents(prediction)
    l_n, l_grains = cv2.connectedComponents(label)    
    
    if p_n > component_limit or l_n > component_limit:
        logger.warning(
            f"Too many components found #predictions:{p_n} #labels:{l_n}. Cropping"
        )
        p_n = min(p_n,component_limit)
        p_grains[p_grains>component_limit] = 0
        l_n = min(l_n,component_limit)
        l_grains[l_grains>component_limit] = 0
    
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
    return _intersection(label,pred)/_union(label,pred)
    
def _iou_from_arr(label_arr,pred_arr):
    pred_mask = _merge(pred_arr)
    label_mask = _merge(label_arr)
    if pred_mask.shape != label_mask.shape or len(label_mask.shape) !=2:
        return np.nan
    return _iou(label_mask,pred_mask)

def f1(precision, recall):
    if precision + recall ==0:
        return np.nan
    return 2*(precision * recall)/(precision + recall)

def calculate_metrics(pred,label,clusters = [0,20,50,100,500,1024**2],component_limit = 500,append_df = False):
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
            "f1":float(f1(p,r))
        }
    
    if append_df:
        return metrics,df
    else:
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

def _threshold_foreground(foreground,thr):
    if thr == 0:
        return np.ones_like(foreground,dtype = np.uint8)
    elif thr == 1:
        return np.zeros_like(foreground,dtype = np.uint8)
    p = np.zeros_like(foreground)
    p[foreground>=thr] = 1
    return np.uint8(p)

def _calc_prec_rec_from_pred(y,p):    

    if (p == 1).all():
        return (0,1)
    elif (p == 0).all():
        return (1,0)
    
    y = np.uint8(y)
    df,_,_ = match_precipitates(p,y)
    return _prec_rec(df)

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
    test_data,
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
    for (test_x,test_y),name in zip(test_data,test_data_names):
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
