import cv2
import matplotlib.patches
import numpy as np
import matplotlib.pyplot as plt
import collections

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

