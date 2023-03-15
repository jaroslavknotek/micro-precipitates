import imageio
import numpy as np
import cv2

import scipy.ndimage

def background_divide(img, foreground_sigma = 1, background_sigma_rel =.15  ):
    background_sigma = int(background_sigma_rel * np.max(img.shape))
    
    img_b =  scipy.ndimage.gaussian_filter(img,foreground_sigma).astype(float)    
    mean_img = np.mean(img_b)
    img_mean_subtracted =  img_b.astype(float)-mean_img
    img_strong_blur = scipy.ndimage.gaussian_filter(img_mean_subtracted,background_sigma)
    
    return  (img_mean_subtracted+ mean_img )/ (img_strong_blur+ mean_img)

def morph_grayscale_close(img,kernel_size = 2):
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size,kernel_size))
    kernel = np.ones((kernel_size,kernel_size))
    
    return cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)

def blur(img,sigma = 1):
    return cv2.GaussianBlur(img,(sigma,sigma),0)
    

def threshold(img,threshold):
    _,th3 =  cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
    return 255-th3


def norm(img):
    img_min = np.min(img)
    img_max = np.max(img)
    if img_min == img_max:
        return img
    
    norm_img = (img - img_min)/(img_max -img_min)
    return (norm_img * 255).astype(np.uint8)


def rotate_image(img, angle_deg):
    
    h,w = img.shape 
    rot_mat = cv2.getRotationMatrix2D((w/2,h/2), angle_deg, 1.0)
    return cv2.warpAffine(img, rot_mat,(w,h) , flags=cv2.INTER_LINEAR)
    
    

def get_bounding_box(mask,padding = 0,append_crop = False):
    """
    returns bounding box top,bottom,left,right pixel coordinates of the object 
    with padding `padding`
    """
    assert padding >=0
    
    yy,xx = np.nonzero(mask)

    nz_l,nz_r  = np.min(xx),np.max(xx)
    nz_t,nz_b = np.min(yy),np.max(yy)
    
    bb_l = np.maximum(0,nz_l-padding)
    bb_r = np.minimum(mask.shape[1],nz_r +padding)
    
    bb_t = np.maximum(0,nz_t-padding)
    bb_b = np.minimum(mask.shape[0],nz_b +padding)
    t,b,l,r =  bb_t,bb_b +1,bb_l,bb_r+1
    if not append_crop:
        return t,b,l,r
    else:
        return (t,b,l,r), mask[t:b,l:r]

def get_shape_contour(binary_img):    
    contours,_ = cv2.findContours(binary_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) == 1, "Multiple or no contour found, expected only one"
    
    c  = contours[0]
    return c[:,0,:]

def _extract_component(component_mask,component_id):
    component_only = component_mask.copy()
    component_only[component_only != component_id] = 0
    component_only[component_only == component_id] = 1
    
    if (component_only == 0).all():
        raise Exception(f"Coponent {component_id} not found")
    
    return component_only.astype(np.uint8)

def extract_component_with_bounding_boxes(binary_img,bb_padding = 2):
    n_found, cmp_mask = cv2.connectedComponents(binary_img)
    components = (_extract_component(cmp_mask,i) for i in range(1,n_found))
    return [get_bounding_box(c,padding=bb_padding,append_crop=True) for c in components]


