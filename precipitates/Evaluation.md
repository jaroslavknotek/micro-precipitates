```python
import pathlib
import sys
sys.path.insert(0,'..')
import precipitates.nn as nn

model_path = pathlib.Path(f"D:/Git_Repos/Models/Streamlit_05-16.h5")

model = nn.build_unet(
    [128,128],
    loss = 'bfl',
    start_filters = 16,
    depth = 3,
    activation = 'elu',)
model.load_weights(model_path)


```

# tady loaduju testovaci sadu


```python
import imageio
import matplotlib.pyplot as plt

img_dir = pathlib.Path(f"D:\Git_Repos\TrainingData\Test_05-16")
img_paths = list(img_dir.glob("*.png"))
imgs = [imageio.imread(img_path) /255  for img_path in img_paths]

for img in imgs:
    plt.imshow(img)
    plt.show()
```

    C:\Users\jan.prochazka\AppData\Local\Temp\ipykernel_30916\3697833436.py:6: DeprecationWarning: Starting with ImageIO v3 the behavior of this function will switch to that of iio.v3.imread. To keep the current behavior (and make this warning disappear) use `import imageio.v2 as imageio` or call `imageio.v2.imread` directly.
      imgs = [imageio.imread(img_path) /255  for img_path in img_paths]
    


    
![png](output_2_1.png)
    



    
![png](output_2_2.png)
    



    
![png](output_2_3.png)
    



    
![png](output_2_4.png)
    



    
![png](output_2_5.png)
    



```python
preds = [nn.predict(model,img) for img in imgs[:1]]
```


```python

for img ,pred,img_path in zip(imgs,preds,img_paths):
    _,(axl,axr) = plt.subplots(1,2)
    plt.suptitle(img_path.stem)
    axl.imshow(img)
    axr.imshow(pred)
    
    
```


    
![png](output_4_0.png)
    



```python
for img ,pred,img_path in zip(imgs,preds,img_paths):
    mask_out = img_path.parent/"masks" / f"{img_path.stem}_mask.png"
    mask_out.parent.mkdir(exist_ok=True)
    imageio.imwrite(mask_out, pred)
```


```python
import precipitates.visualization as visu

for img ,pred in zip(imgs,preds):
    
    img_c = visu.add_contours_morph(img,pred)
    plt.figure(figsize = (12,12))
    plt.imshow(img_c)
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    


    
![png](output_6_1.png)
    



    
![png](output_6_2.png)
    



    
![png](output_6_3.png)
    



    
![png](output_6_4.png)
    



    
![png](output_6_5.png)
    



```python
%pwd

```




    'D:\\Git_repos\\micro-precipitates\\precipitates'


