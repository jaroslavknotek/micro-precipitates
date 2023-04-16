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
DATA_ROOT = "data/20230415"
MODEL_PATH = "model/model-20230416.h5"

if __name__ =='__main__':
    X_train, X_test, y_train, y_test = dataset.read_dataset(DATA_ROOT,IMG_SIZE)

    print("Prepare model")
    model_path = pathlib.Path(MODEL_PATH)
    if model_path.exists():
        print(f"Model already exists")
    else:
        print(f"Existing model not found at {model_path}. Training started")
        model = nn.train(IMG_SIZE,IMG_CHANNELS,model_path, X_train, X_test, y_train, y_test )
        print("Success")

