import cv2
import matplotlib.patches
import numpy as np
import matplotlib.pyplot as plt
import collections

import imageio
import json
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
 
def _show_precipitate_detail(ax,features,img):
    
    radius = features.circle_radius + 2
    t = int(features.circle_y - radius)
    b = t +int(radius*2)
    l = int(features.circle_x - radius)
    r = l +int(radius*2)
    ax.set_xlim((l,r))
    #inverted axes here
    ax.set_ylim((b,t))
    ax.axis('off')
    
    title = f"x:{int(features.circle_x)} y:{int(features.circle_y)} S(px): {int(features.precipitate_area_px)}"
    ax.set_title(title)
    ax.imshow(img)
    
    e = _get_ellipse(features)
    ax.add_patch(e)
    

def plot_precipitate_details(df_features,preds_mask,img,columns=5):
    img=add_contours_morph(img,preds_mask)
    
    rows = int( np.ceil(len(df_features)/columns))
    fig,axs = plt.subplots(rows,columns,figsize=(3*columns,3*rows))
    for ax, features in zip(axs.flatten(), df_features.itertuples()):
        _show_precipitate_detail(ax,features,img)
    fig.suptitle(f"Precipitates Details")
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    return fig

def add_contours_morph(img,mask,contour_width = 1, color_rgb = (255,0,0)):

    if len(img.shape) == 2:
        img_rgb = np.dstack([img]*3)
    else:
        img_rgb = img
        
    contours = mask - cv2.erode(mask,np.ones((3,3))) 
    contours = cv2.dilate(contours,np.ones((contour_width,contour_width)))
    img_rgb[contours==255] = color_rgb
    return img_rgb

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
    
def plot_histograms(df_features,bins=20):
    if 'precipitate_area_um' in df_features.columns: 
        areas = df_features.precipitate_area_um.to_numpy()
    else: 
        areas = df_features.precipitate_area_px.to_numpy()
    
    fig,axs = plt.subplots(2,1)
    
    _,_, bars = axs[0].hist(areas.flatten(),bins=bins)
    # don't show zeros
    axs[0].bar_label(bars, labels=[v if v > 0 else '' for v in bars.datavalues])
    axs[0].set_title(f"Precipitate Area Histogram (Total = {len(areas)})")
    axs[0].set_ylabel("Count")
    axs[0].set_xlabel("$\\mu m$")
    
    _plot_shape_bar(axs[1],df_features)
    return fig

def _get_shape_text(shape_class):
    if shape_class == "shape_irregular":
        return "Irregular"
    elif shape_class == "shape_circle":
        return "Circle"
    else: 
        return 'Needle-like'


def _norm(img):
    m = np.min(img)
    return (img - m)/(np.max(img)-m)

def _save_imgs(eval_path, image_dict):
    for k,v in image_dict.items():
        path = eval_path/f"{k}.png" 
        v = np.uint8(_norm(v)*255)
        imageio.imwrite(path,v)
    
def _plot_loss(H):
    fig,ax = plt.subplots(1,1)
    ax.set_title("Training Loss on Dataset")
    ax.set_xlabel("Epoch #")
    ax.set_ylabel("Loss")

    ax.plot(np.mean(H["train_loss"],axis=1), label="train_loss")
    ax.plot(np.mean(H["val_loss"],axis=1), label="val_loss")
    
    ax.legend(loc="lower left")
    return fig

def save_evaluations(
    eval_root,
    evaluations,
    loss_dict,
    ax_figsize = 10,
    thr = .7
):  
    fig = _plot_loss(loss_dict)
    fig.savefig(eval_root/"loss_figure.png")
    plt.close()
    
    for k,v in evaluations.items():
        eval_path = eval_root/k
        eval_path.mkdir(parents = True,exist_ok = True)
        img_dict = v['images']
        imgs = img_dict| {'pred':np.uint8(img_dict['foreground']>thr)}
        
        # save img
        _save_imgs(eval_path, imgs)
        
        # save fig
        fig = plot_evaluation(imgs,ax_figsize,thr = thr)
        fig.suptitle(k)
        fig.savefig(eval_path/f"{k}_plot.png")
        plt.close()
        
        # json
        json.dump(v['metrics'], open(eval_path/'metrics.json','w'))
        
        
def plot_evaluation(img_dict,ax_figsize = 10,thr = .7):
    n = len(img_dict)
    fig,(axs_imgs,axs_hist) = plt.subplots(2,n,figsize=(ax_figsize*n,ax_figsize*2))
    for ax,(k,img) in zip(axs_imgs,img_dict.items()):
        im = ax.imshow(img,cmap='gray')
        ax.set_title(k)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')


    for ax,(k,img) in zip(axs_hist,img_dict.items()):
        ax.hist(img.flatten(),bins=100)
        ax.set_title(k)
    return fig