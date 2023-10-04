---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: computer-vision
    language: python
    name: .venv
---

```python
%load_ext autoreload
%autoreload 2
```

```python
%cd ../../micro-precipitates/
```

```python
import pathlib
```

```python
# def _exp_see_crops(mask,pred):
    
#     cols = 8
#     zz = list(_zip_pred_label_crops(mask,pred,stride = 64))
#     _,axs = plt.subplots(len(zz)//cols, cols,figsize = (16,len(zz)//4 * 1.2))
#     axs = axs.reshape((-1,2)) 
    
#     for (axl,axr), (m,p) in zip(axs,zz):
#         axl.imshow(m)
#         axr.imshow(p)
#         dwbc = dwbce(m,p)
#         wbc = wbce(m,p)
#         bc = bce(p,p)
#         title = f"BC: {bc:.3f}, WBC: {wbc.3f} DWBC: {dwbc:.3f}"
#         axl.set_title(title)
#     plt.show()

```

```python
import precipitates.evaluation as evaluation

all_models = list(pathlib.Path('/home/jry/test-models/').rglob("*.h5"))    
test_data_dirs = pathlib.Path('data/test/IN')
results = evaluation.evaluate_models(all_models,test_data_dirs,filter_small=True)

```

```python
for res in results:
    result = dict(res)
    del result['img']
    del result['pred']
    del result['mask']
    print(result)
```

```python
fig = evaluation._visualize(results)
```
