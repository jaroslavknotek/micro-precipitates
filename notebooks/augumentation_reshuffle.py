import numpy as np
import pandas as pd
import cv2
import scipy.ndimage as ndi


def synthetize_precipitates(img,mask):
    img_inpainted = get_background(img,mask)
    
    reshufled_img,reshufled_mask = reshuffle_precipitates(img,mask)
    return blend_w_alpha(img_inpainted,reshufled_img),reshufled_mask
        

def reshuffle_precipitates(img,mask):
    df_prec = extract_precipitates(img,mask)
    precipitates,masks = cutout_precipitates(img,mask,df_prec)
    return scatter_precipitates(img,precipitates,masks)    



def extract_precipitates(img,mask):
    mask = cv2.dilate(mask,np.ones((3,3)))
    n_labels,img_labels = cv2.connectedComponents(mask)
    
    data_rows = []
    for i in range(1,n_labels):
        label = np.uint8(img_labels==i)
        
        nz_y,nz_x = np.nonzero(label)
    
        y_min,y_max = np.min(nz_y),np.max(nz_y)
        x_min,x_max = np.min(nz_x),np.max(nz_x)
        
        area_px = np.sum(label)
        mask = label[y_min:y_max,x_min:x_max]
        prec = img[y_min:y_max,x_min:x_max]
        data = (y_min,y_max,x_min,x_max,area_px,prec,mask)
        data_rows.append(data)
        
    return pd.DataFrame(
        data_rows,
        columns = ['y_min','y_max','x_min','x_max','area','prec','mask']
    )

def get_background(img,mask,erode_by = 5,inpaint_by = 15):    
    mask_dilated = cv2.dilate(mask,np.ones([erode_by]*2))
    img_inpainted = cv2.inpaint(np.uint8(img*255),mask_dilated,inpaint_by,cv2.INPAINT_TELEA)
    return img_inpainted/255


def get_random_points(y_shape,x_shape,n):

    yy = np.random.randint(0,y_shape,size=n)
    xx = np.random.randint(0,x_shape,size=n)
    
    return np.vstack([yy,xx])


def cutout_precipitates(img,mask, df_prec):

    precipitates = []
    masks = []
    for row in df_prec.itertuples():
        prec_mask = np.float32(row.mask)
        prec_grad = np.pad(prec_mask,(1,1))
        prec_grad = ndi.gaussian_filter(prec_grad,1)
        prec_grad[prec_grad<.5]=0
        prec_grad = np.nan_to_num(norm(prec_grad))    
        
        prec = np.pad(row.prec,(1,1),mode='reflect')
        prec = np.dstack([prec,prec,prec,prec_grad])

        precipitates.append(prec)        
        masks.append(prec_grad)
    
    return precipitates,masks

def scatter_precipitates(img,precipitates,masks):
    points = get_random_points(
        img.shape[0],
        img.shape[1],
        len(precipitates)
    )
    
    base = np.zeros((img.shape[0],img.shape[1],4))
    synth_mask = np.zeros((img.shape[0],img.shape[1]))
    
    for y,x,p,mask in zip(points[0],points[1], precipitates,masks):
        y_max = min(y + p.shape[0],base.shape[0])
        x_max = min(x + p.shape[1],base.shape[1])
        
        prec_part = p[:y_max-y,:x_max-x]
        prec_rgb = prec_part[:,:,:3]
        prec_alpha = prec_part[:,:,-1]
    
        base_rgb = base[y:y_max,x:x_max,:3]
        base_alpha = base[y:y_max,x:x_max,-1]
    
        rgb_pair = np.array([prec_rgb *prec_alpha[...,None],base_rgb])
        rgb = np.nanmax(rgb_pair,axis=0)
        
        alpha = np.nan_to_num(np.maximum(prec_alpha,base_alpha))
        base[y:y_max,x:x_max] = np.dstack([rgb,alpha])

        mask_part = mask[:y_max-y,:x_max-x]
        synth_mask[y:y_max,x:x_max] +=mask_part
        
    return base,np.uint8(synth_mask>0)


def norm(img):
    img_max = np.max(img)
    img_min = np.min(img)
    return (img -img_min)/(img_max - img_min)

def blend_w_alpha(background,foreground_rgba):
    background = background
    foreground = foreground_rgba[:,:,0]
    alpha = foreground_rgba[:,:,3]

    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    return cv2.add(foreground, background)

