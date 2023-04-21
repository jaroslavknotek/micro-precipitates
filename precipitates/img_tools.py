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
    bbs = [(get_bounding_box(c),c) for c in components]
    return [ ((t,b,l,r),c[t:b,l:r]) for (t,b,l,r),c in bbs]


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
    
def _filter_small(img):
    kernel = np.zeros((4,4),dtype=np.uint8)
    kernel[1:3,:]=1
    kernel[:,1:3]=1
    return cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)