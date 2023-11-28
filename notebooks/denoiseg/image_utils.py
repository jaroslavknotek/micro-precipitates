import numpy as np
import cv2
import imageio


def load_mask(path):
    return load_image(path,norm_img = False)
    
def load_image(path,ensure_2d = True, norm_img = True,crop_to_square = True):
    img = np.squeeze(imageio.imread(path))
   
    if ensure_2d:
        img = _ensure_2d(img)

    if crop_to_square:
        img = img[:img.shape[1]]
    
    if norm_img:
        img = np.float32(norm(img))
    
    return img

def norm(img):
    img_max = np.max(img)
    img_min = np.min(img)
    
    return (img - img_min)/(img_max - img_min)

def denoise_xy(image,noise_percent = .2):


    mask = np.zeros_like(image)
    h,w = mask.shape
    num_sample = int(noise_percent * h*w) 
    mask_indices_r = np.random.randint(2, high = h-2, size = (num_sample))
    mask_indices_c = np.random.randint(2, high = w-2, size = (num_sample))
    mask_indices = np.vstack([mask_indices_r,mask_indices_c])
    
    replacement_offsets = np.random.randint(-2, 3, (2, num_sample))
    replacement_indices = mask_indices + replacement_offsets
    
    mask[mask_indices_r,mask_indices_c] = 1

    noise_from  = image[replacement_indices[0],replacement_indices[1]]

    noise_y = image[None]
    noise_x = np.copy(noise_y)
    noise_x[0, mask_indices_r,mask_indices_c] = noise_from
    return noise_x,noise_y, mask


def label_to_classes(label):
    foreground = label
    background = np.abs(1-foreground)
    border = _get_border(foreground)

    return np.stack([foreground,background,border])


def _get_border(foreground):
    fg_int = np.uint8(foreground)
    kernel = np.ones((3,3))
    eroded = cv2.morphologyEx(fg_int,cv2.MORPH_ERODE,kernel)
    return foreground - eroded

def _ensure_2d(img,ensure_float=True):
    
    match img.shape:
        case (h,w):
            return img
        case (h,w,_):
            return img[:,:,0]
        case _:
            raise ValueError("Unexpected img shape")
