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
```

```python
import pathlib
from tqdm.auto import tqdm

DATA_DIR_ROOT = "data/delisa"
data_dir_root = pathlib.Path(DATA_DIR_ROOT)
```

```python
import imageio
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)

import cv2

def _get_contour(img,dilate_by  = 5):
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilate_by,dilate_by))
    img_d= cv2.dilate(img, element)
    
    return img_d-img

def _draw_contour(img,contour):
    img = _norm(img)
    img_rgb = np.dstack([img]*3)
    img_rgb[:,:,0][contour==np.max(contour)] = 255
    img_rgb[:,:,1][contour==np.max(contour)] = 0
    img_rgb[:,:,2][contour==np.max(contour)] = 0
    return img_rgb

def _draw_label(img,label):
    contour = _get_contour(label)
    return _draw_contour(img,contour)

def _get_prec_circles(mask,circle_padding = 0):
    n, prec_mask = cv2.connectedComponents(mask)
    
    
    prec_mask[prec_mask>0] =1
    prec_mask = np.uint8(prec_mask)
    contours, hierarchy = cv2.findContours(prec_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    circles = []
    for contour in contours:
        (x,y),radius = cv2.minEnclosingCircle(contour)
        circles.append(((x,y),radius + circle_padding))
        
    return circles



def _plot_circles(ax,circles):
    for (x,y),r in circles:
        c = plt.Circle((x,y), r, color='yellow',fill=False,linewidth=3)
        ax.add_patch(c)
        
def _draw_labels_circles(img,label,ax=None,figsize=(20,20)):    
    if ax is None:
        _,ax = plt.subplots(1,1,figsize=figsize)
    ax.imshow(img,cmap='seismic',vmin=0)
    circles = _get_prec_circles(label,circle_padding = 5)
    _plot_circles(ax,circles)


    
    
    
img = imageio.imread('/home/jry/source/jaroslavknotek/micro-precipitates/data/Delisa-castice/otagovany/DELISA LTO_08Ch18N10T_pricny rez_nulty stav_TOP_BSE_03.tif')
label_raw = imageio.imread('/home/jry/source/jaroslavknotek/micro-precipitates/data/Delisa-castice/otagovany/DELISA LTO_08Ch18N10T_pricny rez_nulty stav_TOP_BSE_03_OZNACENI_CASTIC.tif')

img,label_raw = [_norm(_crop_bottom_bar(img)) for img in [img,label_raw]]


#only non grayscale
label = ~np.bitwise_and(label_raw[:,:,0] == label_raw[:,:,1],label_raw[:,:,1] == label_raw[:,:,2]) != np.zeros((label_raw.shape[0],label_raw.shape[1]))
label = label.astype(int)
label_255 = (label*255).astype(np.uint8)


label_large_raw = _crop_bottom_bar(imageio.imread("data/Delisa-castice/otagovany/DELISA LTO_08Ch18N10T_pricny rez_nulty stav_TOP_BSE_03_binary_label.png"))
label_255 =label_large_raw

_,(ax_img,ax_label) = plt.subplots(1,2,figsize = (20,10))
ax_img.imshow(img)
ax_label.imshow(_draw_label(img,label_255))

```

```python
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

#     conmat = np.array([
#         [tp, fp], 
#         [fn,tn]
#     ])
#     _print_confusion_matrix(m)
    
    precision = np.sum(~df['label_id'].isna() & ~df['pred_id'].isna()) / grains_pred
    
    recall =  np.sum(~df['label_id'].isna() & ~df['pred_id'].isna()) / grains_label
    
    
    if not include_df:
        df = None
        
    return precision,recall,df
    
def _print_confusion_matrix(conmat):
    df_cm = pd.DataFrame(conmat, index = ["D","ND"],columns=["D","ND"])
    sn.heatmap(df_cm, annot=True,fmt = 'd') # font size
    
```

# Find threshold

```python
import cv2
from tqdm.auto import tqdm

def plot_prec_rec(prec,rec, thresholds=None):
    plt.plot(prec,label = "precision")
    plt.plot(rec,label = "recall")
    
    best_t = np.argmin(np.abs(prec-rec))
    plt.title(f"{thresholds[best_t]} - {xx.T[best_t]}")
    plt.legend()
    if thresholds is not None:
        plt.xticks(ticks=np.arange(len(thresholds)), labels=thresholds)
    plt.show()


thresholds = np.arange(0,80,5)
res = [ compare(_threshold(_blur(img,3),t),label_255) for t in tqdm(thresholds)]
xx = np.array([(p,r) for p,r,_ in res]).T
plot_prec_rec(xx[0],xx[1],thresholds)
plt.show()

```

```python
thresholds = np.arange(40,100,5)
res = [ compare(_threshold(_blur(img,5),t),label_255) for t in tqdm(thresholds)]
xx = np.array([(p,r) for p,r,_ in res]).T
plot_prec_rec(xx[0],xx[1],thresholds)
```

# Normalize background

**TODO** - read the gonzales about it

```python
import scipy.ndimage

a = scipy.ndimage.gaussian_filter1d(img[500] ,3)
x= a - np.mean(a)
y = scipy.ndimage.gaussian_filter1d(x,150)

z = (x + np.mean(a)) /(y + np.mean(a))

print(np.max(z),np.min(z))

plt.plot(x)
plt.plot(y)
plt.plot(z*20)
#plt.ylim((-100,100))
```

```python

    
#img_backg_noise_div[np.isnan(img_backg_noise_div)] = 0
imgs = [
    img,
    _background_divide(img),
    _background_divide(img,foreground_sigma=3)
]


_,axs = plt.subplots(1,len(imgs),figsize = (30,10))
_ = [ax.imshow(img) for ax,img in zip(axs,imgs)]
```

```python
drawn_ctrs = [_draw_label(img,label_255) for img in imgs]
_,axs = plt.subplots(1,len(imgs),figsize = (30,10))
_ = [ax.imshow(img) for ax,img in zip(axs,drawn_ctrs)]
```

```python
np.vstack([thresholds,xx ]).T
```

```python
img_bck_normed =  _norm(_background_divide(img))
thresholds = np.arange(50,121,5)
res = [ compare(_threshold(img_bck_normed,t),label_255) for t in tqdm(thresholds)]
xx = np.array([(p,r) for p,r,_ in res]).T

plot_prec_rec(xx[0],xx[1],thresholds)
plt.show()
```

# Morphology

- threshold based on histogram **\[MANUAL\]**
- morphology based on size -> **\[MANUAL\]**

```python
threshold_95recall = 85
img_pred_95 = _threshold(img_bck_normed,threshold_95recall)

_,axs=plt.subplots(1,2,figsize=(20,10))

axs[0].hist(img.flatten(),bins=np.arange(255))
axs[0].axvline(threshold_95recall,c='r')

axs[1].imshow(img_pred_95)

#run compare based on morphology
```

```python

```

```python

```

```python

img_compare = np.dstack([img_pred_95 ,label_255 ,_norm(img_bck_normed)//2])
plt.figure(figsize=(10,10))
plt.imshow(img_compare[:200,:200])
```

# IDEA I can run piecewise comparion


Here, I will use open and then compare

```python
elem_size = 2
elem = np.ones((elem_size,elem_size))

img_opened = cv2.morphologyEx(img_pred_95,cv2.MORPH_OPEN,elem)

compare(img_opened,label_255),compare(img_pred_95,label_255)


# much worse recall
```

# NEXT STEPS

- two types:
  - use original image
  - apply background normalization
- different thresholds
- different morph size

$|types| \times |thresholds| \times |morph sizes| = 2 \times 5 \times 3 = 30$


#  NOPE Subtract blurred stuff


```python
def exp_subtract_blurred_stuff():
    bl = scipy.ndimage.gaussian_filter(img,5)
    
    res_sh = np.zeros(img.shape)
    res_sh[img <= bl] =1
    
    
    imgs = [img,bl,res_sh]
    _,axs = plt.subplots(len(imgs),1,figsize = (10,40))
    
    [ax.imshow(img[:200,:200]) for ax,img in zip(axs,imgs)]
    
exp_subtract_blurred_stuff()
```

```python
def _gs_morph(img,kernel_size = 2):
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    kernel = np.ones((kernel_size,kernel_size))
    
    return cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)


thresholds = np.arange(30,91,5)
res = [ compare(_threshold(_gs_morph(img),t),label_255) for t in tqdm(thresholds)]
xx = np.array([(p,r) for p,r,_ in res]).T

plot_prec_rec(xx[0],xx[1],thresholds)
plt.show()

# imgs = [morphed_trh, morphed,img]
# cimgs = [_draw_label(img,label_255) for img in imgs]
# _,axs = plt.subplots(len(imgs),2,figsize = (20,40))   

# axs_c1 = axs[:,0]
# axs_c2 = axs[:,1]
# [ax.imshow(img[:200,:200]) for ax,img in zip(axs_c1,imgs)]
# [ax.imshow(img[:200,:200]) for ax,img in zip(axs_c2,cimgs)]
```

```python
plt.figure(figsize=(10,10))
plt.imshow(img[320:450,720:870],cmap='gray')
plt.show()

img_m = _gs_morph(img, 3)

imgs = [img,img_m]
cimgs = [_draw_label(img,label_255) for img in imgs]
_,axs = plt.subplots(len(imgs),2,figsize = (20,20))   

axs_c1 = axs[:,0]
axs_c2 = axs[:,1]
[ax.imshow(img[320:450,720:870],cmap='seismic',vmin = 0) for ax,img in zip(axs_c1,imgs)]
[ax.imshow(img[320:450,720:870]) for ax,img in zip(axs_c2,cimgs)]
```

```python
# thresholds = np.arange(80,141,5)
# n_g_img=_norm(_gs_morph(img,3))
# res = [ compare(_threshold(n_g_img,t),label_255) for t in tqdm(thresholds)]
# xx = np.array([(p,r) for p,r,_ in res]).T

# plot_prec_rec(xx[0],xx[1],thresholds)
```

# TODO incorporate moments
- rotation normalization
  - Careful about symmetry
  
# NEXT steps

Find better thresholds
Don't use grayscale morhpology -> use threshold and normal one



```python
import matplotlib.pyplot as plt
import img_tools
from pathlib import Path

import precipitates         



data_root = Path('data/Delisa-castice/')
images_paths = list(data_root.rglob('DELISA*/**/*.tif'))

imgs = list(map(precipitates.load_microscope_img,images_paths))


test_img_path = [ p for p in images_paths if p.name == 'DELISA LTO_08Ch18N10T_pricny rez_nulty stav_TOP_BSE_03.tif'][0]
test_img = precipitates.load_microscope_img(test_img_path)
threshold = 120
                                               
    
test_shapes = precipitates.extract_raw_mask(test_img,threshold)
plt.imshow(test_shapes)

```

```python
import pandas as pd
from tqdm.auto import tqdm
    
def get_precipitate_description_dataframe(img,threshold):
    mask,shapes = precipitates.identify_precipitates(img,threshold)
    precipitates_features = list(map(precipitates.extract_features,shapes))
    precipitates_classes = [ precipitates.classify_shape(f) for f in precipitates_features]

    df = pd.DataFrame(precipitates_features)
    df['shape_class'] = precipitates_classes
    total_px =  img.shape[0]*img.shape[1]
    df['precipitate_area_ratio'] = df['precipitate_area_px']/total_px
                                       
    return df,mask

threshold = 100
dfs_masks = [ get_precipitate_description_dataframe(img,threshold) for img in tqdm(imgs,desc="processing images")]

```

```python
import numpy as np

def _show_precipitate_detail(ax,features,img):
    
    radius = features.circle_radius + 2
    t = int(features.circle_y - radius)
    b = t +int(radius*2)
    l = int(features.circle_x - radius)
    r = l +int(radius*2)
    title = f"x:{int(features.circle_x)} y:{int(features.circle_y)}"
    ax.set_title(title)
    ax.imshow(img,cmap='gray', vmin=0,vmax=255)
    ax.set_xlim((l,r))
    ax.set_ylim((t,b))
    ax.axis('off')
    e = _get_ellipse(features)
    ax.add_patch(e)
    
    

# columns = 10
# rows = min(int( np.ceil(len(df)/columns)),200)

# _,axs = plt.subplots(rows,columns,figsize=(20,3*rows))
# for ax, features in zip(axs.flatten(), df.itertuples()):
#     _show_precipitate_detail(ax,features,test_img)
# plt.show()
```

```python
import collections


def _get_shape_text(shape_class):
    if shape_class == "shape_irregular":
        return "Irregular"
    elif shape_class == "shape_circle":
        return "Circle"
    else: 
        return 'Needle-like'

def _plot_area_histogram(ax,df,bins = 100):
    ax.set_title("Area Distribution")
    ax.hist(df.precipitate_area_ratio,bins = bins)
    ax.set_ylabel("Number of precipitates in the bin")
    ax.set_xlabel("Histogram bins")

def _plot_shape_bar(ax,df):
    labels, values = zip(*collections.Counter(df['shape_class']).items())
    indexes = np.arange(len(labels))
    width=.8
    bar_colors = list(map(_get_shape_color,labels))
    texts = list(map(_get_shape_text,labels))
    
    ax.set_title("Shape Distribution")
    ax.bar(indexes, values, width,color = bar_colors)
    ax.set_xticks(indexes, texts)
    ax.set_ylabel("Number of precipitates")
    ax.set_xlabel("Shapes")
    

# df,_ =dfs_masks[0]
# img_path = images_paths[0]

# _,axs = plt.subplots(1,2,figsize = (8,4))
# plt.suptitle(img_path.name)
# _plot_area_histogram(axs[0],df)
# _plot_shape_bar(axs[1],df)

# plt.tight_layout()
```

```python

import matplotlib.patches
import os
import imageio

def _get_shape_color(shape_class):
    if shape_class == "shape_irregular":
        return "magenta"
    elif shape_class == "shape_circle":
        return "#00FF00"
    else: 
        return 'red'

def _get_ellipse(features):
    
    
    deg = features.ellipse_angle_deg    
    x = features.ellipse_center_x
    y =features.ellipse_center_y
    width = features.ellipse_width_px 
    height = features.ellipse_height_px

    c = _get_shape_color(features.shape_class)
    return matplotlib.patches.Ellipse(
        (x,y),
        width,
        height,
        angle = deg,
        fill=False,
        edgecolor=c,
        lw=2,
        alpha = .5
    )

def _plot_precipitates(ax,df,img):
    ax.imshow(img,cmap='gray')
    for features in df.itertuples():
        e = _get_ellipse(features)
        ax.add_patch(e)

output_path = Path("output")
for img,path,(df,mask) in tqdm(zip(imgs,images_paths,dfs_masks),total=len(imgs),desc="Plotting img_data"):
    img_output = output_path/path.stem
    os.makedirs(img_output,exist_ok=True)
    
    df.to_csv(img_output/"data.csv",index=False)
    
    _,axs = plt.subplots(1,2,figsize = (8,4))
    
    plt.suptitle(path.name)
    _plot_area_histogram(axs[0],df)
    _plot_shape_bar(axs[1],df)
    
    plt.tight_layout()    
    plt.savefig(img_output/"dist.png")   
    plt.close()
    
    _,ax  = plt.subplots(1,1,figsize= (10,10))
    _plot_precipitates(ax, df,img)
    ax.set_title(path.name)
    plt.savefig(img_output/"highlight.png")
    plt.close()
    
    mask_255 = np.uint8(mask) *255
    imageio.imwrite(img_output/"mask.png",mask_255)
    imageio.imwrite(img_output/"img.png",img)
    
    
    #detail
    columns = 10
    rows = min(int( np.ceil(len(df)/columns)),20)
    _,axs = plt.subplots(rows,columns,figsize=(3*columns,3*rows))
    for ax, features in zip(axs.flatten(), df.itertuples()):
        _show_precipitate_detail(ax,features,img)
    plt.suptitle("Detail of Precipitates")
    plt.savefig(img_output/"details.pdf")
    plt.tight_layout()
    plt.close()
    
    
```

```python
for img,path,(df,mask) in zip(imgs,images_paths,dfs_masks):
    print(path.name)
    _,ax  = plt.subplots(1,1,figsize= (10,10))
    _plot_precipitates(ax, df,img)
    ax.set_title(path.name)
    plt.show()
    
```

```python

```

```python
for path,img in zip(images_paths,imgs):
    plt.imshow(img)
    plt.title(path.name)
    print(path.name)
    plt.show()
```
