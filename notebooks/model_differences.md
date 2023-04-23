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
%cd ../../micro-precipitates

import sys
sys.path.insert(0,'python')
```

# Running model

```python
import app
import nn

import pathlib

new_model_pars = {
    "MODEL_PATH": "model/model-20230406.h5",
    "IMG_FOLDER" : 'data/20230405/not_labeled',
    "TRAIN_FOLDER":'data/20230405/labeled',
    "OUTPUT_FOLDER" : "output/distributions/20230405"
}

old_model_pars = {
    "MODEL_PATH": "model/model-20230328.h5",
    "IMG_FOLDER"  : 'data/20230328/not_labeled',
    "TRAIN_FOLDER":'data/20230328/labeled',
    "OUTPUT_FOLDER" : "output/distributions/20230328"
}

combined_model_pars = {
    "MODEL_PATH" : "model/model-20230407-combined.h5",
    "IMG_FOLDER" : 'data/20230407-combined/not_labeled',
    "TRAIN_FOLDER":'data/20230407-combined/labeled',
    "OUTPUT_FOLDER" : "output/distributions/20230407-combined"
}

pars_20230415 = {
    "MODEL_PATH" : "model/model-20230416.h5",
    "TRAIN_FOLDER":'data/20230415/labeled',
    "IMG_FOLDER" : 'data/20230415/not_labeled',
    "OUTPUT_FOLDER" : "output/distributions/20230416"
}
pars_20230417_morp_opening = {
    "MODEL_PATH":"model/model-20230417-morp_opening.h5",
    "TRAIN_FOLDER":"data/20230417-morp_opening/labeled",
    "IMG_FOLDER" :"data/20230417-morp_opening/not_labeled",
    "OUTPUT_FOLDER":"output/distributions/20230416-morp_opening"
}


param_dicts = [
    # new_model_pars,
    # old_model_pars,
    combined_model_pars,
    pars_20230415,
    pars_20230417_morp_opening
]

def predict_with_model(dict_params):
    args = app._parse_args([
            '--imgfolder', dict_params['IMG_FOLDER'], 
            '--modelpath', dict_params['MODEL_PATH'],
            '--outputfolder',dict_params['OUTPUT_FOLDER']])


    IMG_EXTS = ['.png','.jpg','.jpg','.tif','.tiff']
    img_paths = [file for file in pathlib.Path(args.imgfolder).rglob("*.*") if file.suffix in IMG_EXTS]
    model = nn.load_model(args.modelpath)

    out_dir = pathlib.Path(args.outputfolder)
    out_dir.mkdir(parents=True,exist_ok=True)

    app._run_batch(img_paths,model,out_dir,notify_progress=True)
    
```

```python

for dp in param_dicts:
    predict_with_model(dp)

```

# Gatherring common

```python
import pathlib

def load_comparing_imgs(param_dicts):
    
    outs_paths= []
    sets = []
    for pd in param_dicts:
        output_folder = pd['OUTPUT_FOLDER']
        masks_paths = list(pathlib.Path(output_folder).rglob('*_mask.*'))
        masks_names = set([f.name for f in masks_paths])
        
        outs_paths.append(masks_paths)
        sets.append(masks_names)
    
    common = set.intersection(*sets)
    
    common_paths = [[ p.parent for p in paths if p.name in common ] for paths in outs_paths]
    return common_paths

#param_dicts = [new_model_pars,old_model_pars] 
titles = [ pathlib.Path(dp['MODEL_PATH']).stem for dp in param_dicts]
common_paths =  load_comparing_imgs(param_dicts)
```

```python
import imageio
import matplotlib.pyplot as plt

for roots in  zip(*common_paths):
    imgs = [imageio.imread(list(root.glob('*_contoured.png'))[0]) for root in roots]
    assert len(imgs) == len(titles)
    
    _,axs = plt.subplots(1,len(imgs),figsize = (len(common_paths)*10,11))
    plt.suptitle(roots[0].name)
    
    for ax,img,title in zip(axs,imgs,titles):
        ax.imshow(img,cmap='gray')
        ax.set_title(title)
        ax.axis('off')
        
    plt.tight_layout()

    plt.savefig(f"detail_{roots[0].name}.pdf")
    
    
```

```python
import cv2
import pandas as pd 
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def _extract_grain_mask(labels,grain_id):
    grain = labels.copy()
    grain[labels == grain_id] = 1
    grain[labels != grain_id] = 0
    
    if (grain == 0).all():
        raise Exception(f"Grain {grain_id} not found")
    
    return grain

    
def _pair_grains(predicted,label):
    p_n, p_grains = cv2.connectedComponents(predicted)
    l_n, l_grains = cv2.connectedComponents(label)
    pairs = []
    for p_grain_id in range(1,p_n):
        p_grain_mask = _extract_grain_mask(p_grains,p_grain_id)
        
        l_grain_id = np.max(p_grain_mask*l_grains)
        
        
        if l_grain_id!=0:
            l_grain_mask = _extract_grain_mask(l_grains,l_grain_id)
        else:
            l_grain_id = None
            l_grain_mask = None
            
        pairs_rec = (
            p_grain_id,
            l_grain_id,
            p_grain_mask,
            l_grain_mask,
        )
        pairs.append(pairs_rec)
    
    used_labels = [ l[1] for l in pairs]
    false_negatives = [fn for fn in np.arange(1,l_n) if fn not in used_labels]
    for fn in false_negatives:
        l_grain_mask = _extract_grain_mask(l_grains,fn)
        pairs.append((None,fn,None,l_grain_mask))
    return pd.DataFrame(pairs,columns = ['pred_id','label_id','pred_mask','label_mask'])    


def compare(predicted,label,include_df = False):
    df =  _pair_grains(predicted,label)
    
    grains_pred = df['pred_id'].max()
    grains_label = df['label_id'].max()

    # todo check that pairs are not twice
    tp = len(df[ ~df['label_id'].isna() & ~df['pred_id'].isna()])
    fp = len(df[ df['label_id'].isna() & ~df['pred_id'].isna()])
    fn = len(df[ ~df['label_id'].isna() & df['pred_id'].isna()])
    tn = 0 #len(df[ ~df['label_id'].isna() & df['pred_id'].isna()])

    precision = np.sum(~df['label_id'].isna() & ~df['pred_id'].isna()) / grains_pred
    
    recall =  np.sum(~df['label_id'].isna() & ~df['pred_id'].isna()) / grains_label
    
    
    if not include_df:
        return precision,recall
    return precision,recall,df
    
def _print_confusion_matrix(conmat):
    df_cm = pd.DataFrame(conmat, index = ["D","ND"],columns=["D","ND"])
    sn.heatmap(df_cm, annot=True,fmt = 'd') # font size
```

```python
from tqdm.auto import tqdm
import cv2

def _filter_small(img):
    kernel = np.zeros((4,4),dtype=np.uint8)
    kernel[1:3,:]=1
    kernel[:,1:3]=1
    return cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)


test_root = list(pathlib.Path("data/test").glob('*'))
test_names = [dir_path.name for dir_path in test_root]
test_imgs=   [precipitates.load_microscope_img(dir_path/'img.png') for dir_path in test_root]
test_masks_raw=   [imageio.imread(dir_path/'mask.png') for dir_path in test_root]

test_masks_openning = [ _filter_small(mask) for mask in test_masks_raw ]

tests = list(zip(test_names,test_imgs,test_masks_openning))

model_results = []
for d in tqdm(param_dicts,total=len(param_dicts),desc='generating results'):
    model = nn.compose_unet(128,128,3)
    model.load_weights(d['MODEL_PATH'])
    
    model_results.append([(name,img,_filter_small(gt), _filter_small(nn.predict(model,img))) for name,img,gt in tests])

    
import pandas as pd
import tensorflow as tf
results_rows = []
for d, res in zip(param_dicts, model_results):
    model_name = pathlib.Path(d['MODEL_PATH']).stem
    for exp_name, img,gt,pred in res:
        m = tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])
        m.update_state(gt/255>.5,pred/255>.5)
        iou = m.result().numpy()
        row = (
            model_name,
            exp_name,
            iou,
            img,
            gt,
            pred
        )
        results_rows.append(row)
        
columns = [
    "model_name",
    "test_name",
    "iou",
    "img",
    "gt",
    "pred"
]

df_all = pd.DataFrame(results_rows,columns=columns)


```

```python
pr = np.array([compare(_filter_small(row.pred),_filter_small(row.gt)) for row in df_all.itertuples()])
df_all['precision'] = pr[:,0]
df_all['recall'] = pr[:,1]
df_all['f1'] = 2*(df_all['precision'] * df_all['recall'])/(df_all['precision'] + df_all['recall'])

df = df_all[["model_name","test_name","iou","f1"]]
```

```python
p = df.pivot(index='test_name', columns='model_name', values='iou')
p.style.background_gradient(axis=1)  
```

```python
p = df.pivot(index='test_name', columns='model_name', values='f1')
p.style.background_gradient(axis=1)  
```

```python
df = df_all.sort_values(by=['model_name','test_name'])
df_in = df[df['test_name'].str[-2:] == 'IN']
fig,axs = plt.subplots(len(df_in),2,figsize=(18,len(df_in)*9))
for axrow,row in zip(axs,df_in.itertuples()):
    model_test = f"{row.model_name} - {row.test_name} - IoU:{row.iou:.5} - F1:{row.f1:.5} - P:{row.precision:.5} - R:{row.recall:.5}"
    axrow[0].set_title(model_test)
    axrow[0].imshow(row.img)
    im = axrow[1].imshow((row.gt.astype(float) - row.pred)/255,cmap='seismic')
    fig.colorbar(im, orientation='vertical')

```
