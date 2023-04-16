import pathlib
import imageio
import shutil
import cv2
from tqdm.auto import tqdm
import numpy as np
import itertools

import sklearn.model_selection

def dump_dataset(dataset_root, out_dir ,img_size = 128):
    imgs,masks = _read_dataset(dataset_root)
    out_dir = pathlib.Path(out_dir)    
    t_imgs = tqdm(imgs,desc='Augumenting and cropping input images',total=len(masks))
    # data = np.array(list(_get_crops_dataset_iter(t_imgs,masks,img_size)))
    data = _get_crops_dataset_iter(t_imgs,masks,img_size)
    
    zeros = 6

    for i,(img,mask) in enumerate(data):
        suffix = str(i).zfill(zeros)
        mask_path = out_dir/"mask"/f"img_{suffix}.png"
        mask_path.parent.mkdir(exist_ok=True,parents=True)
        imageio.imwrite(mask_path,mask)

        img_path = out_dir/"img"/f"img_{suffix}.png"
        img_path.parent.mkdir(exist_ok=True,parents=True)
        imageio.imwrite(img_path,img)


def read_dataset(dataset_root,img_size):
    imgs_masks = _read_image_mask_pair(dataset_root)
    
    imgs_masks = tqdm(
        imgs_masks,
        desc='Augumenting and cropping input images',
        total=len(masks))

    crops = ( iter(augument_and_crop(img,mask)) for img,mask in imgs_mask)
    chained = itertools.chain(crops)

    data = np.fromiter(crops, dtype = (np.uint8,(2,img_size,img_size)))
    
    X = np.array([ _ensure_three_chanels(i) for i in data[:,0]])
    y = data[:,1]
    return X,y[:,:,:,np.newaxis] 
#     X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2, random_state=42)
#     y_train = y_train[...,np.newaxis]
#     y_test = y_test[...,np.newaxis]
#     
#     return X_train, X_test, y_train, y_test


def _to_graylevel(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img,cv2.COLOR_RGBA2GRAY)
    return img
    
def _to_binary(img):
    
    gs_img = _to_graylevel(img)

    return np.int8(gs_img >=255)*255
#    _,thr = cv2.threshold(gs_img,128,255,cv2.THRESH_BINARY)
#    return thr 
    

def _read_image_mask_pair(dataset_root):
    """
    This method assumes that each image is named img.png and is located
    in a separate folder along with label named mask.png
    """
    folder_path = pathlib.Path(dataset_root)
    imgs_p = list(folder_path.rglob("img.png"))
    masks_p = [img.parent/'mask.png' for img in imgs_p]
    
    imgs = map(imageio.imread, imgs_p)
    imgs_gray = map(_to_graylevel,imgs)
    masks = map(imageio.imread,masks_p)
    masks_bin = map(_to_binary,masks)

    return list(zip(imgs_gray,masks_bin))


def _rotate_image(image, angle):
    h,w = image.shape
    image_center = w/2,h/2
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(image, rot_mat,(w,h), flags=cv2.INTER_CUBIC)
    

def augument_and_crop(img,mask,size,seed= 4567):
    np.random.seed = seed
    padded_size = int(np.ceil(np.sqrt(2)*size) +1)    
    
    stride = 10
    shape = (padded_size,padded_size)
    imgs = np.lib.stride_tricks.sliding_window_view(img,shape)[::stride,::stride].reshape((-1,*shape))
    masks = np.lib.stride_tricks.sliding_window_view(mask,shape)[::stride,::stride].reshape((-1,*shape))
    n=len(imgs)
    #rotate
    angles =np.random.uniform(0,360,size=n)
    imgs_r= (_rotate_image(i,a) for i,a in zip(imgs,angles))
    masks_r= (_rotate_image(i,a) for i,a in zip(masks,angles))
    
    #flip
    flip_probability = .5
    flips = np.random.uniform(0,1,size=n)<flip_probability
    imgs_f= ((np.flip(i) if f else i) for i,f in zip(imgs_r,flips))
    masks_f= ((np.flip(i) if f else i) for i,f in zip(masks_r,flips))
    
    # TODO scale ?
    
    # crop
    c = padded_size//2
    t = c-size//2
    b = t +size
    l = t
    r = b    
    
    img_crops = np.array(list(imgs_f),dtype=np.uint8)[:,t:b,l:r]
    mask_crops= np.array(list(masks_f),dtype=np.uint8)[:,t:b,l:r]
    
    crops =  np.stack([img_crops,mask_crops])
    return np.swapaxes(crops,0,1)
 
def _ensure_three_chanels(img):
    if len(img.shape) != 3 or img.shape[2]!=3:
        return np.dstack([img]*3)
    return img
