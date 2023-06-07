```python
%load_ext autoreload
%autoreload 2
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload
    

**Import packages and set up parameters**


```python
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
```

**Select training folder and build datasets**


```python
train_data = "D:\\Git_Repos\\TrainingData\\Train_05-17\\"
train_data = pathlib.Path(train_data)
train_ds,val_ds = ds.prepare_datasets(
    train_data,
    crop_stride=fixed_config['crop_stride'],
    crop_shape = fixed_config['crop_shape'],
    filter_size = fixed_config['filter_size'],
    cache_file_name=f'.cache-{crop_size}',
    generator=False
)
```

    DEBUG:prec:Found 2 images
    DEBUG:PIL.PngImagePlugin:STREAM b'IHDR' 16 13
    DEBUG:PIL.PngImagePlugin:STREAM b'tIME' 41 7
    DEBUG:PIL.PngImagePlugin:b'tIME' 41 7 (unknown)
    DEBUG:PIL.PngImagePlugin:STREAM b'IDAT' 60 8192
    DEBUG:PIL.PngImagePlugin:STREAM b'IHDR' 16 13
    DEBUG:PIL.PngImagePlugin:STREAM b'tIME' 41 7
    DEBUG:PIL.PngImagePlugin:b'tIME' 41 7 (unknown)
    DEBUG:PIL.PngImagePlugin:STREAM b'IDAT' 60 8192
    DEBUG:PIL.PngImagePlugin:STREAM b'IHDR' 16 13
    DEBUG:PIL.PngImagePlugin:STREAM b'tIME' 41 7
    DEBUG:PIL.PngImagePlugin:b'tIME' 41 7 (unknown)
    DEBUG:PIL.PngImagePlugin:STREAM b'IDAT' 60 8192
    DEBUG:PIL.PngImagePlugin:STREAM b'IHDR' 16 13
    DEBUG:PIL.PngImagePlugin:STREAM b'tIME' 41 7
    DEBUG:PIL.PngImagePlugin:b'tIME' 41 7 (unknown)
    DEBUG:PIL.PngImagePlugin:STREAM b'IDAT' 60 8192
    DEBUG:prec:Found examples: 841
    DEBUG:prec:Sizes. Train: 43.0, Val: 11.0. Batch: 32
    

**Preview training images**


```python
for img,mask in train_ds:
    _,(axl,axr) = plt.subplots(1,2)
    axl.imshow(img[0])
    axr.imshow(mask[0])
    
    plt.show()
```


    
![png](output_6_0.png)
    



    
![png](output_6_1.png)
    



    
![png](output_6_2.png)
    



    
![png](output_6_3.png)
    



    
![png](output_6_4.png)
    



    
![png](output_6_5.png)
    



    
![png](output_6_6.png)
    



    
![png](output_6_7.png)
    



    
![png](output_6_8.png)
    



    
![png](output_6_9.png)
    



    
![png](output_6_10.png)
    



    
![png](output_6_11.png)
    



    
![png](output_6_12.png)
    



    
![png](output_6_13.png)
    



    
![png](output_6_14.png)
    



    
![png](output_6_15.png)
    



    
![png](output_6_16.png)
    



    
![png](output_6_17.png)
    



    
![png](output_6_18.png)
    



    
![png](output_6_19.png)
    



    
![png](output_6_20.png)
    



    
![png](output_6_21.png)
    



    
![png](output_6_22.png)
    



    
![png](output_6_23.png)
    



    
![png](output_6_24.png)
    



    
![png](output_6_25.png)
    



    
![png](output_6_26.png)
    


**Define training parameters**


```python
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
```

**Training**


```python
import precipitates.train

model_path = pathlib.Path(f"D:/Git_Repos/Models/Streamlit_05-17.h5")
       
precipitates.train.run_training_w_dataset(
    train_ds,
    val_ds,
    paras,
    model_path
)
```

    DEBUG:prec:Will checkpiont model to D:\Git_Repos\Models\Streamlit_05-17.h5
    

    Epoch 1/100
    27/43 [=================>............] - ETA: 27s - loss: 0.0146 - io_u_1: 0.9973WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 4300 batches). You may need to use the repeat() function when building your dataset.
    

    WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 4300 batches). You may need to use the repeat() function when building your dataset.
    

    WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 11 batches). You may need to use the repeat() function when building your dataset.
    

    WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 11 batches). You may need to use the repeat() function when building your dataset.
    

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,io_u_1
    

    WARNING:tensorflow:Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,io_u_1
    

    WARNING:tensorflow:Can save best model only with val_loss available, skipping.
    

    WARNING:tensorflow:Can save best model only with val_loss available, skipping.
    

    43/43 [==============================] - 47s 1s/step - loss: 0.0146 - io_u_1: 0.9973
    
