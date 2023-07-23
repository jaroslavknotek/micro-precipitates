import imageio
import numpy as np
import cv2
import pandas as pd
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
    
    

def get_bounding_box(mask):
    """
    returns bounding box top,bottom,left,right pixel coordinates of the object 
    """
    
    yy,xx = np.nonzero(mask)

    nz_l,nz_r  = np.min(xx),np.max(xx)
    nz_t,nz_b = np.min(yy),np.max(yy)
    
    bb_l = np.maximum(0,nz_l)
    bb_r = np.minimum(mask.shape[1],nz_r )
    
    bb_t = np.maximum(0,nz_t)
    bb_b = np.minimum(mask.shape[0],nz_b)
    return  bb_t,bb_b +1,bb_l,bb_r+1

def get_shape_contour(binary_img):    
    padding= 2
    img = np.pad(binary_img,padding)
    contours,_ = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) == 1, "Multiple or no contour found, expected only one"
    
    c  = contours[0]
    return c[:,0,:]  -padding

def _extract_component(component_mask,component_id):
    component_only = component_mask.copy()
    component_only[component_only != component_id] = 0
    component_only[component_only == component_id] = 1
    
    if (component_only == 0).all():
        raise Exception(f"Coponent {component_id} not found")
    
    return component_only.astype(np.uint8)

def extract_component_with_bounding_boxes(binary_img):
    n_found, cmp_mask = cv2.connectedComponents(binary_img)
    components = (_extract_component(cmp_mask,i) for i in range(1,n_found))
    bbs = ((get_bounding_box(c),c) for c in components)
    return [ ((t,b,l,r),c[t:b,l:r]) for (t,b,l,r),c in bbs]


def _extract_grain_mask(labels,grain_id):
    grain = labels.copy()
    grain[labels == grain_id] = 1
    grain[labels != grain_id] = 0
    
    if (grain == 0).all():
        raise Exception(f"Grain {grain_id} not found")
    
    return grain

def img2crops(img, stride, shape):
    assert len(img.shape)==2,f"Image must be 2d. Is: {len(img.shape)}"
    slider = np.lib.stride_tricks.sliding_window_view(img,shape)
    strider =  slider[::stride,::stride]
    return strider.reshape((-1,*shape))[:,:,:]

def _pair_grains(predicted,label,cap = 500):
    p_n, p_grains = cv2.connectedComponents(predicted)
    l_n, l_grains = cv2.connectedComponents(label)
    
    
    if cap is not None:
        p_n = min(cap,p_n)
        p_grains [p_grains >cap] = 0
        
        l_n = min(cap,l_n)
        l_grains [l_grains >cap] = 0
    
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

    
def _print_confusion_matrix(conmat):
    df_cm = pd.DataFrame(conmat, index = ["D","ND"],columns=["D","ND"])
    sn.heatmap(df_cm, annot=True,fmt = 'd') # font size
    

def filter_small(mask,size_limit):
    
    if size_limit == 0:
        return mask
    
    old_dtype = mask.dtype
    mask = np.uint8(mask)
    n,lbs = cv2.connectedComponents(mask)
    base = np.zeros_like(mask,dtype=np.uint8)
    for i in range(1,n):
        component = np.uint8(lbs==i)
        size = np.sum(component)
        if size >= size_limit:
            base = base | component
    return base.astype(old_dtype)

def _cut_square(img,top,left,square_size):
    
    h,w,*_ = img.shape
    
    
    bottom = min(top+square_size,h-1)
    top_refitted = bottom - square_size
    
    right = min(left+square_size,w-1)
    left_reffited = right - square_size
    
    return img[top_refitted:bottom,left_reffited:right]   
    
def _get_stride_shape(img_shape,stride):
    stride_shape = np.ceil(np.array(img_shape[:2])/stride)*stride//stride +1
    return np.int32(stride_shape) 

def cut_to_squares(img,square_size,stride):
    
    bordered_shape = np.array(img.shape[:2])-square_size
    stride_shapes = _get_stride_shape(bordered_shape,stride)
    
    stride_y = (np.arange(stride_shapes[0])*stride).astype(int)
    stride_x = (np.arange(stride_shapes[1])*stride).astype(int)
    
    yx = [(y,x) for y in stride_y for x in stride_x]
    return np.array([_cut_square(img,y,x,square_size) for y,x in yx])
    
def decut_squares(squares,stride,orig_shape):
    
    square_shape = squares.shape[1:]
    square_size = square_shape[0]
    bordered_shape = np.array(orig_shape)-square_size
    stride_shape = _get_stride_shape(bordered_shape,stride)
    squares_2d = squares.reshape( (*stride_shape,*square_shape))
    
    base = np.zeros(orig_shape)
    cont = np.zeros(orig_shape)
    
    stride_shapes = _get_stride_shape(orig_shape,stride)
    row_strides = np.arange(stride_shapes[0]) * stride
    col_strides = np.arange(stride_shapes[1]) * stride
    h,w,*_ = orig_shape
    for square_row,row_stride in zip(squares_2d,row_strides):
        
        for square,col_stride in zip(square_row,col_strides):
            crop_height = min(square_size,h-row_stride)
            crop_width = min(square_size,w-col_stride)
            
            h_s,w_s,*_ = square.shape
            square_crop = square[h_s - crop_height:,w_s - crop_width:]
            
            base[
                row_stride:row_stride + crop_height,
                col_stride:col_stride+crop_width
            ] += square_crop
            
            cont[
                row_stride:row_stride + crop_height,
                col_stride:col_stride+crop_width
            ] += 1
        
    
    return (base/cont).astype(squares.dtype)