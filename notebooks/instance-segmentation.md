---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.0
  kernelspec:
    display_name: cv_torch
    language: python
    name: cv_torch
---

```python
import sys
sys.path.insert(0,'..')
import precipitates.nnet as nnet

```

```python
import pathlib
import precipitates.dataset as ds
data_20230623_root = pathlib.Path('../data/20230623/labeled/')

data_test_root = pathlib.Path('../data/test/')

result_root = pathlib.Path('../results-instance')

named_data_test= ds.load_img_mask_pair(data_test_root,append_names=True)

data_20230623 = ds.load_img_mask_pair(data_20230623_root)

f"{len(data_20230623)=},{len(named_data_test[0])=}"
```

```python
import matplotlib.pyplot as plt
import numpy as np
import cv2

x,y = data_20230623[0]


def _get_masks_and_bboxes_from_y(y,min_sum = 4):
    y_max = np.max(y)
    if y_max ==0:
        return [],[]
        
    y8bit = np.uint8(y/y_max*255)
    
    num,labels = cv2.connectedComponents(y8bit)

    masks = []
    bboxes = []
    for label_id in range(15,num):
        mask = np.uint8(labels==label_id)
        if np.sum(mask) <min_sum:
            continue
            
        nz = np.nonzero(mask)
        top,bottom,left,right = nz[0].min(), nz[0].max(), nz[1].min(), nz[1].max()        
        bbox = left,top,right+1,bottom+1
        masks.append(mask)
        bboxes.append(bbox)
        
    return masks,bboxes


masks,bboxes = _get_masks_and_bboxes_from_y(y)

#plt.imshow(  masks[np.argmax()] )
```

```python
id_largest = np.argmax(np.sum(masks,axis=(1,2)))
bbox = bboxes[id_largest]

plt.imshow(masks[id_largest])
plt.xlim((bbox[0],bbox[2]))
plt.ylim((bbox[1],bbox[3]))
```

```python
import time

# start = time.time()
# instance_segm_data_triplets = [(x,*_get_masks_and_bboxes_from_y(y)) for  x,y in data_20230623]
# end = time.time()
# end-start
```

```python
import traceback

import albumentations as A
import cv2
import torch

def _get_augumentation(crop_size,interpolation=cv2.INTER_CUBIC):
    #bbox_format = 'pascal_voc'
    safe_padding = 1.5
    cs_padded = [int(crop_size *safe_padding)]*2
    return A.Compose([
        A.PadIfNeeded(*cs_padded),
        A.RandomCrop(*cs_padded),
        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        #A.augmentations.transforms.GaussNoise(.05),
        A.Rotate(limit=45, interpolation=interpolation),
        A.ElasticTransform(
            p=.5,
            alpha=10, 
            sigma=120 * 0.1,
            alpha_affine=120 * 0.1,
            interpolation=interpolation
        ),
        A.RandomCrop(crop_size,crop_size),
        
    ]
    )

class InstanceDataset(torch.utils.data.Dataset):
    def __init__(self, data_pairs,crop_size = 128):
        self.data_pairs = data_pairs
        self.transforms = _get_augumentation(crop_size)
        self.crop_size = crop_size

    def __getitem__(self, idx):

        x,y = self.data_pairs[idx]
        if self.transforms is not None:
            transformed = self.transforms(
                image = x,
                mask =y,
            )
            x = transformed['image']
            y = transformed['mask']



        masks,bboxes = _get_masks_and_bboxes_from_y(y)

        category_ids = np.ones((len(masks),))    
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        category_ids = torch.as_tensor(category_ids, dtype=torch.int64)
        iscrowd = torch.zeros((len(bboxes),), dtype=torch.int64)

        image_id = torch.tensor([idx])
        if len(bboxes) > 0:
            area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
            masks = torch.as_tensor(np.array(masks)==1, dtype=torch.uint8)
            bboxes = torch.as_tensor(np.array(bboxes), dtype=torch.float32)
        else:
            area = torch.empty((0,1), dtype=torch.float32) 
            masks = torch.empty((0,1,self.crop_size,self.crop_size), dtype=torch.uint8)
            bboxes = torch.empty((0,4), dtype=torch.float32)
            
        target = {
            "boxes" : bboxes,
            "labels" : category_ids,
            "masks" : masks,
            "image_id" : image_id,
            "area" : area,
            "iscrowd" : iscrowd
        }
        
        xx = np.stack([x]*3)
        img = torch.as_tensor(xx,dtype=torch.float32)
        
        return img, target
        
    def __len__(self):
        return len(self.data_pairs)

import time
start = time.time()

instance_dataset = list(InstanceDataset(data_20230623,crop_size=512))

print(len(data_20230623), len(instance_dataset))
for x,targets in instance_dataset:
    plt.imshow(x[0])
    print(targets)
    plt.show()
    break
    
end = time.time()

print(f"{end - start}")
```

```python
import torch
import torchvision

num_classes = 2

weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights 
model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
    weights = weights,
    progress=True,
    num_class = num_classes
)
```

```python
orig_pred = model.roi_heads.mask_predictor

in_channels = orig_pred.conv5_mask.in_channels
dim_reduced = orig_pred.mask_fcn_logits.in_channels

predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
    in_channels,
    dim_reduced,
    num_classes
)
    
#model.roi_heads.mask_predictor = predictor
```

```python
import sys
# this stuff https://github.com/pytorch/vision/tree/main/references/detection
sys.path.insert(0,'/home/knotek/source/vision/references/detection/')


import utils

p = np.random.permutation(len(data_20230623))

val_percent = .3
val_size = int(len(p)*val_percent)

d = np.array(data_20230623,dtype=object)
train_data = d[p][:-val_size]
val_data = d[p][-val_size:]
assert len(train_data) + len(val_data) == len(data_20230623)

train_dataset = InstanceDataset(train_data,512)
train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=2, shuffle=True,
    collate_fn=utils.collate_fn
)

val_dataset = InstanceDataset(val_data,512)
val_data_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=False,
    collate_fn=utils.collate_fn
)

```

```python

from engine import train_one_epoch, evaluate
```

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)

epochs = range(100)
for epoch in epochs:
    train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=10)
    lr_scheduler.step()
    print('next epoch')
    #evaluate(model, val_data_loader, device=device)    
```

# Test Eval

```python
#x_3ch = np.random.rand(3,512,512).astype(np.float32)
xx = [ x for x,t in instance_dataset if len(t['boxes']) >1][0]

xx.shape
```

```python
from torchvision.utils import draw_bounding_boxes


model.eval()


batch = xx[None,...]
batch = (batch).to(device)
with torch.no_grad():
    p = model(batch)

#torchvision.models.detection.maskrcnn_resnet50_fpn_v2(

#model_name = 'MASKRCNN_RESNET50_FPN_V2'


pred = p[0]
#.cpu().detach()
boxes = pred['boxes']#.cpu().detach().numpy()
labels = pred['labels']


ndarr = batch[0].mul(255).clamp_(0, 255).to('cpu', torch.uint8)
#ndarr.shape
if len(boxes) > 0:
    x_bb = draw_bounding_boxes(ndarr ,boxes,labels=[str(l.cpu().detach().numpy()) for l in labels])
    plt.figure(figsize=(10,10))
    plt.imshow(x_bb[0].to('cpu'))

```

```python
from torchvision.utils import draw_segmentation_masks

masks = pred['masks'][:,0]>.5
img_masks = draw_segmentation_masks(x_bb,masks)
plt.figure(figsize=(10,10))
plt.imshow(img_masks[0].to('cpu'),cmap='gray')
```

```python

```
