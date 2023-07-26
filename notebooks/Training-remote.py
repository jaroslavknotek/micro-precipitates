#%load_ext autoreload
#%autoreload 2

import sys
sys.path.insert(0,'..')
import pathlib
import numpy as np
import imageio
import matplotlib.pyplot as plt
import precipitates.dataset as ds

import logging
logging.basicConfig(level=logging.DEBUG)

crop_size=128

fixed_config = {
    "crop_stride":32,
    "crop_shape":(crop_size,crop_size),
    "filter_size":0
}


model_path = pathlib.Path(f"D:/Git_Repos/Models/Streamlit_07-26.h5")
train_data = "D:\Git_Repos\TrainingData\Streamlit_RIP"
train_data = pathlib.Path(train_data)
train_imgs = list(train_data.rglob("**/img.png"))
print(f"Train images count: {len(train_imgs)}")

for img in train_imgs:
    strimg=str(img)
    print(str(img).replace(str(train_data),'').replace('img.png','')[1:-1])

train_ds,val_ds = ds.prepare_datasets(
    train_data,
    crop_stride=fixed_config['crop_stride'],
    crop_shape = fixed_config['crop_shape'],
    filter_size = fixed_config['filter_size'],
    cache_file_name=f'.cache-{crop_size}',
    generator=True
)

args = {
    'crop_stride':32,
    'patience':5,
    # 'crop_stride':{'values': [256]},
    # 'patience':{'values': [0]},
    #'loss':{'values': ['bc','wbc-1-2','wbc-2-1','wbc-5-1','bfl']}, # 'dwbc', 'wbc-1-2' removed
    'loss':'bfl', # 'dwbc', 'wbc-1-2' removed
    'filter_size':0,
    'cnn_filters':16,
    'cnn_depth':3,
    'cnn_activation':'elu',
    'crop_size':128
}  

class my_dict:
    pass

paras = my_dict()
paras.__dict__ = args

import precipitates.train

       
precipitates.train.run_training_w_dataset(
    train_ds,
    val_ds,
    paras,
    model_path
)