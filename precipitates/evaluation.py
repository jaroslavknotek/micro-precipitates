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


def _pair_grains(*params):
    return it._pair_grains(*params)

def _intersection(a,b):
    if a is None or b is None:
        return np.nan
    else:
        return a*b
    
def _union(a,b):
    assert a is not None or b is not None
        
    if a is None:
        a = np.zeros(b.shape)
    if b is None:
        b = np.zeros(a.shape)
    
    return np.sum(a+b>=1)

def _category(row,clusters):
    
    vals = [x for x in [row.label_area_px,row.pred_area_px] if x is not None]
    assert len(vals) >0
    min_size = np.nanmin(vals)
    for a,b in itertools.pairwise(clusters):
        if a<=min_size<b:
            return clusters.index(a)
    assert False

def prec_rec(df):
    grains_pred = df['pred_id'].max()
    grains_label = df['label_id'].max()

    # todo check that pairs are not twice
    tp = len(df[ ~df['label_id'].isna() & ~df['pred_id'].isna()])
    fp = len(df[ df['label_id'].isna() & ~df['pred_id'].isna()])
    fn = len(df[ ~df['label_id'].isna() & df['pred_id'].isna()])
    tn = 0 #len(df[ ~df['label_id'].isna() & df['pred_id'].isna()])

    precision = np.sum(~df['label_id'].isna() & ~df['pred_id'].isna()) / grains_pred
    recall =  np.sum(~df['label_id'].isna() & ~df['pred_id'].isna()) / grains_label
    return precision,recall

def _merge(masks):
    masks = [mask for mask in masks if mask is not None]
    return np.uint8(np.sum(masks,axis=0)>0)
   
def _iou(label,pred):
    m = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])
    m.update_state(label,pred)
    return m.result().numpy()

def _iou_from_arr(label_arr,pred_arr):
    pred_mask = _merge(label_arr)
    label_mask = _merge(pred_arr)
    if pred_mask.shape != label_mask.shape or len(label_mask) !=2:
        return np.nan
    return _iou(label_mask,pred_mask)

def _f1(precision, recall):
    return 2*(precision * recall)/(precision + recall)

def _calculate_metrics(pred,label,clusters = [0,50,500,1024**2]):
    df =  _pair_grains(pred,label)
    
    df['pred_area_px'] = [ np.sum(x) for x in df.pred_mask]
    df['label_area_px'] = [ np.sum(x) for x in df.label_mask]
    df['size_category'] = [ _category(row,clusters) for row in df.itertuples()]
    
    cat_size = {u:df[df.size_category == u] for u in df.size_category.unique()}
    cat_size[-1] = df
    
    metrics = {}
    for u,sc_df in cat_size.items():
        iou =_iou_from_arr(
            sc_df.pred_mask.to_numpy(),
            sc_df.label_mask.to_numpy()
        )
        p,r = prec_rec(sc_df)
        metrics[int(u)] = {
            "iou":float(iou),
            "precision":float(p),
            "recall":float(r),
            "f1":float(_f1(p,r))
        }
    
    return metrics

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


def evaluate(model, img, ground_truth,filter_small):
    img = _norm(img)
    pred = nn.predict(model,img)    
    if filter_small:
        pred = it._filter_small(pred)
        
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
    filter_small = False):
    
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
                filter_small
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
