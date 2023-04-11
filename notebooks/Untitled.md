---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python

```

```python

import pathlib

import imageio

img_paths = pathlib.Path('D:/OneDrive/UJV/Halodova Patricie - MCA_M4/111/a').glob('*.tif')
import matplotlib.pyplot as plt

import shutil
for img_path in img_paths:
    plt.imshow(imageio.imread(img_path),cmap='gray')
    plt.show()
    mask_path = img_path.parent/'output'/img_path.name
    #print(mask_path,mask_path.exists())
    print(img_path.stem)
    # create folder with img_path.stem
    # copy img_path to <folder>/img.tif
    # copy mask_path t mask.tif
```

```python

```
