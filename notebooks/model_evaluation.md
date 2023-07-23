---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.6
  kernelspec:
    display_name: palivo_general
    language: python
    name: palivo_general
---

```python
%load_ext autoreload
%autoreload 2
%cd ../../micro-precipitates

import pathlib
import sys
sys.path.insert(0,'../precipitates')
import precipitates.nn as nn
```

```python
TEST_ROOT =  pathlib.Path('data/test/IN/DELISA LTO_08Ch18N10T_pricny rez_nulty stav_TOP_BSE_09/')
MODEL_PATH = pathlib.Path('model/model-20230328.h5')
```

# Evaluate 

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

```
