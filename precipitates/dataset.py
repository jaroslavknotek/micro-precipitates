import numpy as np
import matplotlib.pyplot as plt
import pathlib
import itertools
import precipitates.img_tools as img_tools
import torch
import random
import logging
import albumentations as A
import cv2
import imageio
import itertools
from skimage.measure import label
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger("prec")

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        images_mask_tuples, 
        crop_size,repeat = 1,
        generate_weight_map = True
    ):
        self.images_mask_tuples = images_mask_tuples
        self.transform = _get_augumentation(crop_size)
        self.crop_size = crop_size
        self.repeat = repeat
        self.generate_weight_map = generate_weight_map
        
    def __len__(self):
        return len(self.images_mask_tuples) * self.repeat
    
    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError()
        
        idx = idx % len(self.images_mask_tuples)
        image,label = self.images_mask_tuples[idx]

        has_label = label is not None
        if not has_label:
            label = np.zeros_like(image)


        transformed = self.transform(image=image ,mask=label)

        image = transformed['image']

        noise_y = np.copy(image)[None,...]
        noise_x = np.copy(noise_y)

        mask, mask_ind,replacement_ind = _get_mask(self.crop_size)

        noise_x[:, mask_ind[0],mask_ind[1]] = noise_x[:,replacement_ind[0],replacement_ind[1]]

        foreground = transformed['mask']
        background = np.abs(foreground -1)
        border = _get_border(foreground)

        wc = {
            0: 1, # background
            1: 5  # objects
        }
        weight_map = unet_weight_map(foreground, wc)
        if not self.generate_weight_map:
            weight_map = np.ones(weight_map.shape)
        
        y = np.stack([noise_y[0],foreground,background,border])
        x =  np.concatenate([noise_x]*3,axis=0)
        has_label = np.expand_dims(np.array([has_label]),axis=(1,2,3))

        return (
            x,
            y, 
            mask[None,...], 
            has_label,
            np.expand_dims(weight_map,axis=0)
        )
    

def prepare_train_val_dataset(
    dataset_array,
    crop_size,
    apply_weight_map = True,
    val_size = .2,
    batch_size = 32,
    repeat = 1
):
    
    total_dataset_len = len(dataset_array)
    val_count = int(total_dataset_len * val_size)
    train_count = total_dataset_len -  val_count 

    train_dataset = Dataset(
        dataset_array[:-val_count],
        crop_size,
        repeat=repeat,
        generate_weight_map = apply_weight_map
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = Dataset(
        dataset_array[-val_count:],
        crop_size,
        repeat=repeat, 
        generate_weight_map = apply_weight_map
    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader,val_dataloader



def unet_weight_map(y, wc=None, w0 = 10, sigma = 5):
    
    labels = label(y)
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[:,:,i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=2)
        d1 = distances[:,:,0]
        d2 = distances[:,:,1]
        w = w0 * np.exp(-1/2*((d1 + d2) / sigma)**2) * no_labels
        
        if wc:
            class_weights = np.zeros_like(y)
            for k, v in wc.items():
                class_weights[y == k] = v
            w = w + class_weights
        else:
            w = w +1
    else:
        w = np.zeros_like(y)
    
    return w


    
def _get_border(foreground):
    fg_int = np.uint8(foreground)
    kernel = np.ones((3,3))
    eroded = cv2.morphologyEx(fg_int,cv2.MORPH_ERODE,kernel)
    return foreground - eroded

def _get_mask(patch_size,percent = .2):
    mask = np.zeros((patch_size,patch_size))
    num_sample = int(percent * patch_size * patch_size) 
    mask_indices = np.random.randint(2, patch_size-2, (2, num_sample))
    replacement_offsets = np.random.randint(-2, 3, (2, num_sample))
    replacement_indices = mask_indices + replacement_offsets
    
    ind_x,ind_y = mask_indices
    mask[ind_x,ind_y] = 1
    return mask,mask_indices,replacement_indices


def _filter_not_used_denoise_paths(dataset_root,denoise_root):
    def _clean(p):
        return ''.join([ s if s.isalnum() else '_' for s in p])
    used = {_clean(f.parent.stem) for f in dataset_root.rglob("img.png")}
    all_denoise = list(denoise_root.rglob("*.tif"))
    return [p for p in all_denoise if _clean(p.stem) not in used]
    
def load_with_denoise(dataset_root,denoise_root = None):
    if denoise_root is None:
        denoised = []
    else:
        denoise_path = _filter_not_used_denoise_paths(dataset_root,denoise_root)
        denoised = [load_image(d) for d in denoise_path]

    segmentation_pairs = load_img_mask_pair(dataset_root)
    denoised_pairs = zip(denoised,[None]*len(denoised))
    return list(itertools.chain(denoised_pairs,segmentation_pairs))




def prepare_datasets(
    dataset_root,
    denoise_root = None,
    crop_size=128,
    batch_size = 32,
    seed = 123,
    validation_split_factor = .2,
    filter_size = 0,
    repeat = 100,
    interpolation=cv2.INTER_CUBIC):

    dataset_array = load_with_denoise(dataset_root,denoise_root)
    
    logger.info(f"Loading Dataset from {dataset_root}")
    
    np.random.seed = seed
    dataset_array = load_img_mask_pair(dataset_root,filter_size)
    train_size,val_size = _get_train_test_size(len(dataset_array), validation_split_factor)
    
    ds_images = tf.data.experimental.from_list(dataset_array)
    
    dss = [ds_images.take(train_size),ds_images.skip(train_size)]
    return [_prep_ds(ds,repeat,batch_size,crop_size,interpolation) for ds in dss]


def load_img_mask_pair(dataset_root,append_names = False):
    named_pairs = list(_get_img_mask_iter(dataset_root))
    
    pairs = [(img,mask) for _,img,mask in named_pairs]
    if append_names:
        names = [r[0] for r in named_pairs]
        return pairs,names
    else:
        return pairs

def _get_augumentation(crop_size,interpolation=cv2.INTER_CUBIC):
    
    # make space for augumentations -> prevents mirroring
    crop_size_padded = int(crop_size*1.5)
    return A.Compose([
        A.PadIfNeeded(crop_size_padded,crop_size_padded),
        A.RandomCrop(crop_size_padded,crop_size_padded),
        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.augmentations.transforms.GaussNoise(.05),
        A.Rotate(limit=45, interpolation=interpolation),
        A.ElasticTransform(
            p=.5,
            alpha=10, 
            sigma=120 * 0.1,
            alpha_affine=120 * 0.1,
            interpolation=interpolation
        ),
        A.CenterCrop(crop_size,crop_size),
    ])

def _norm_float(img):
    img_f = np.float32(img)
    img_min = np.min(img_f)
    img_max = np.max(img_f)
    return (img_f-img_min)/(img_max-img_min)
    

def _ensure_2d(img):
    match img.shape:
        case (h,w):
            return img
        case (h,w,_):
            return img[:,:,0]

def load_image(img_path,normalize=True,ensure_two_dim = True,ensure_square = True):    
    img = imageio.imread(img_path)
    
    if ensure_square:
        m = np.min(list(img.shape[:2]))
        img = img[:m,:m]
    
    if ensure_two_dim:
        img = _ensure_2d(img)
    
    if normalize:
        img = _norm_float(img)
    
    return img

def _get_img_mask_iter(dataset_root):
    dataset_root = pathlib.Path(dataset_root)
    for img_root in dataset_root.glob('*'):
        try:
            img = load_image(img_root/'img.png')
            mask = load_image(img_root/'mask.png')
            mask = (mask>0).astype(mask.dtype)
            yield img_root,img,mask
        except FileNotFoundError as e:
            logger.warning(f"Skipped {img_root}. Didn't find both files. Error {e}")
