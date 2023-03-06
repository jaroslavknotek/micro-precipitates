---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: palivo
    language: python
    name: palivo
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

import cv2

def _norm(img):
    img_min = np.min(img)
    img_max = np.max(img)
    if img_min == img_max:
        return img
    
    norm_img = (img - img_min)/(img_max -img_min)
    return (norm_img * 255).astype(np.uint8)

def _crop_bottom_bar(img,bar_height = 120):
    return img[:-bar_height]

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
    
    

img = imageio.imread('/home/jry/source/jaroslavknotek/micro-grain/data/delisa/Delisa-castice/otagovany/DELISA LTO_08Ch18N10T_pricny rez_nulty stav_TOP_BSE_03.tif')
label_raw = imageio.imread('/home/jry/source/jaroslavknotek/micro-grain/data/delisa/Delisa-castice/otagovany/DELISA LTO_08Ch18N10T_pricny rez_nulty stav_TOP_BSE_03_OZNACENI_CASTIC.tif')

img,label_raw = [_norm(_crop_bottom_bar(img)) for img in [img,label_raw]]


#only non grayscale
label = ~np.bitwise_and(label_raw[:,:,0] == label_raw[:,:,1],label_raw[:,:,1] == label_raw[:,:,2]) != np.zeros((label_raw.shape[0],label_raw.shape[1]))
label = label.astype(int)
label_255 = (label*255).astype(np.uint8)


label_large_raw = _crop_bottom_bar(imageio.imread("data/label_large_raw.png"))
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

# def _crop_nz(img):
#     coords = np.argwhere(img)
#     x_min, y_min = coords.min(axis=0)
#     x_max, y_max = coords.max(axis=0)
#     return img[x_min:x_max+1, y_min:y_max+1]
    
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

def _blur(img,sigma = 1):
    return cv2.GaussianBlur(img,(sigma,sigma),0)
    

def _threshold(img,threshold):
    _,th3 =  cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
    return 255-th3

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
import scipy.ndimage

def _background_divide(img, foreground_sigma = 1, background_sigma_rel =.15  ):
    background_sigma = int(background_sigma_rel * np.max(img.shape))
    
    img_b =  scipy.ndimage.gaussian_filter(img,foreground_sigma).astype(float)    
    mean_img = np.mean(img_b)
    img_mean_subtracted =  img_b.astype(float)-mean_img
    img_strong_blur = scipy.ndimage.gaussian_filter(img_mean_subtracted,background_sigma)
    
    background_normed =  (img_mean_subtracted+ mean_img )/ (img_strong_blur+ mean_img)
    
    return background_normed
    
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
thresholds = np.arange(80,141,5)
res = [ compare(_threshold(_norm(_gs_morph(img,3)),t),label_255) for t in tqdm(thresholds)]
xx = np.array([(p,r) for p,r,_ in res]).T

plot_prec_rec(xx[0],xx[1],thresholds)
```

```python
thresholds = np.arange(80,141,5)
res = [ compare(_threshold(_norm(_gs_morph(_background_divide(img),3)),t),label_255) for t in tqdm(thresholds)]
xx = np.array([(p,r) for p,r,_ in res]).T

plot_prec_rec(xx[0],xx[1],thresholds)
```

```python
thresholds = np.arange(80,141,5)
res = [ compare(_threshold(_norm(_gs_morph(_background_divide(img),4)),t),label_255) for t in tqdm(thresholds)]
xx = np.array([(p,r) for p,r,_ in res]).T

plot_prec_rec(xx[0],xx[1],thresholds)
```

```python
_,ax = plt.subplots(1,1,figsize=(20,20))
ax.imshow(_background_divide(img),cmap='gray',vmin=0)
_plot_circles(ax,_get_prec_circles(label_255))(x,y),radius = cv.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
```

```python
_draw_labels_circles(_background_divide(img),label_255)
```

```python
_,ax = plt.subplots(1,1,figsize=(20,20))
ax.imshow(_gs_morph(_background_divide(img),3),cmap='seismic',vmin=0)
_plot_circles(ax,_get_prec_circles(label_255))
```

```python
threshold = 120
test_shapes = _threshold(_norm(_gs_morph(_background_divide(img),3)),threshold)
plt.imshow(test_shapes)
```

```python
precision,recall,df = compare(test_shapes, label_255,include_df=True)
```

```python
def _analyse_shape():
    pass


def _get_contour(binary_img):
    
    contours,hierarchy = cv2.findContours(binary_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) == 1, "Multiple or no contour found, expected only one"
    
    return contours[0]
    
def _get_bb(contour,padding_px = 0):    
    x,y,w,h = cv2.boundingRect(contour)
    
    top = max(y-padding_px,0)
    bottom = min(y+h + padding_px,img.shape[0])
    left = max(x-padding_px,0)
    right = min(x+w + padding_px, img.shape[1])
    
    return (top,bottom,left,right),contour
    
def _plot_shape(ax,img,mask,padding_px = 25):
    contour = _get_contour(mask)
    res = img*mask
    (t,b,l,r) =_get_bb(contour,padding_px=padding_px)
    pred = img[t:b,l:r]
    pred_m = mask[t:b,l:r] 
    # TODO, add pred_m as a channel

    (cx,cy),r = cv2.minEnclosingCircle(contour)
    y = cy - t
    x = cx - l
            
    ax.imshow(pred)
    circle = plt.Circle(
        (x,y),
        r, 
        color='#FF00FF',
        fill=False,
        linewidth=3)
    
    ax.add_patch(circle)

    
def _analyse_geometry(immask):
    
    if mask is None:
        return None,None
    mask = mask.astype(np.uint8)
    
    contour = _get_contour(mask)
    
    (cx,cy),r = cv2.minEnclosingCircle(contour)
    area = cv2.contourArea(contour)
    
    return PrecipitationGeometry(
        contour = contour,
        area = area,
        circle_radius = r,
        center_x = cx,
        center_y = cy
    )
    
    
from dataclasses import dataclass

@dataclass
class PrecipitationGeometry:
    """Geometric properties of a precipitation contour"""
    contour: object # it's there for convinience, it should not be here afterwards
    area: float
    circle_radius:float
    center_x:int
    center_y:int
    
    def _circle_area(self) -> float:
        return np.pi*self.circle_radius**2
    
    def circle_similarity(self) -> float:
        return self.area / self._circle_area()

    

def _process_results(df, img):
    
    for idx, row in df.iterrows():
        if row["pred_mask"] is not None:    
            # MAYBE I would like to extract and threshold stuff
            geometry = _analyse_geometry(row["pred_mask"])
            
            return
            
            print(area, circle_area, area/circle_area) 
            
            _,ax = plt.subplots(1,1)
            
            
            plt.show()
            return
    
    
    circles = _get_prec_circles(precipitates_mask)

_process_results(df,img)

```
