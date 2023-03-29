import sys
sys.path.insert(0,"python")

import pathlib
import python.dataset as dataset

import pathlib
import python.nn as nn


import python.precipitates as precipitates
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt





IMG_SIZE = 128
IMG_CHANNELS = 3
DATA_ROOT = "data/labeled"
MODEL_PATH = "model/model-prec-1.h5"



def _run_batch(img_paths,model):
    for img_path in tqdm(img_paths):
        img = precipitates.load_microscope_img(img_path)
        
        if len(img.shape) ==3:
            img = img[:,:,0]
            
        if img.shape != (1024,1024):
            print(img.shape)
            continue
            
        
        preds_mask = nn.predict(model,img)
        plt.imshow(img)
        plt.show()
        plt.imshow(preds_mask)
        return 
    
        name = img_path.name.replace(' ','_')
        img_3ch = np.dstack([img]*3)
        imageio.imwrite(out_dir/name.replace(".tif","_original.png"),img_3ch)
        
        z = np.zeros(img.shape)
        mask_4ch = np.dstack([preds_mask,z,z,preds_mask])
        #imageio.imwrite(out_dir/name.replace(".tif","_mask.png"),mask_4ch)

if __name__ =='__main__':
    X_train, X_test, y_train, y_test = dataset.read_dataset(DATA_ROOT,IMG_SIZE)

    print("Prepare model")
    model_path = pathlib.Path(MODEL_PATH)
    if model_path.exists():
        print(f"Loading existing model {model_path}")
        model = nn.load_model(model_path)
    else:
        print(f"Existing model not found at {model_path}. Training started")
        model = train_model(IMG_SIZE,IMG_CHANNELS,model_path)
        
    

    print("Reading data")
    tif_paths = list(pathlib.Path("data").rglob("*.tif"))
    labeled_img_paths = list(pathlib.Path("data/labeled").rglob("img.png"))
    labeled_img_paths_names = [p.parent.name for p in labeled_img_paths ]
    test_img_paths = [p for p in tif_paths if p.stem not in labeled_img_paths_names]

    print("Running prediction")
    _run_batch(test_img_paths,model)
