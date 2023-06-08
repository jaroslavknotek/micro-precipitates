---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: micro-precipitates
    language: python
    name: micro-precipitates
---

```python
%load_ext autoreload
%autoreload 2
%cd ../../micro-precipitates

import pathlib
import sys
sys.path.insert(0,'../precipitates')
import precipitates
```

```python
TEST_ROOT =  pathlib.Path('D:/Git_Repos/TrainingData/Train_05-22/008_SMMAG_x300k_411/')
MODEL_PATH = pathlib.Path('D:/Git_Repos/Models/Streamlit_05-16.h5')
```

# Evaluate 

```python
import numpy as np
np.max(img_test)
```

```python
res = nn.predict(model,img_test)
plt.imshow(res)

```

```python
import precipitates.evaluation as evaluation
import imageio
import precipitates.nn as nn
import matplotlib.pyplot as plt


img_test = imageio.imread(TEST_ROOT/"img.png")
gt_test = imageio.imread(TEST_ROOT/"mask.png")//255

model = nn.load_model(MODEL_PATH)

(img,ground_truth,pred,metrics_res) = evaluation.evaluate(model, img_test,gt_test)

fig,ax_row= plt.subplots(1,4,figsize=(20,4))
evaluation._visualize_pairs(
    ax_row,
    img_test,
    ground_truth,
    pred,
    metrics_res,
    MODEL_PATH.stem
)
```

```python
plt.imshow(pred)
```

```python
metrics_res

```
