# Identify and classifiy precipitates from SEM imagery
---

# Get repository

[Clone repository](https://git-scm.com/docs/git-clone) using the following command:

```
git clone https://github.com/jaroslavknotek/micro-precipitates.git
```

Or if you want to [use ssh certificate](https://linuxkamarada.com/en/2019/07/14/using-git-with-ssh-keys/#.ZCQJppixXmE)

```
git clone git@github.com:jaroslavknotek/micro-precipitates.git 
```

# Preparation

Locate into the repository

```
cd micro-precipitates
```

## Setup Virtual Environemnt

This repo uses `conda`.
```
conda env create --name <env-name> 
conda activate <env-name>

conda install opencv
pip install -r requirements.txt
```
# Inference

```
import precipitates.nnet as nnet
import precipitates.dataset as ds
model = torch.load(<model-path?)
image = ds.load_image(<img-path>)

res = nnet.predict(model,image)

segmentation_mask = np.uint8(res['foreground']>THR)
```

# Training

[TODO](notebooks/training.md)
