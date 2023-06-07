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

## Virtual environemnt

Create virtual environment 
```
python -m virtualenv .venv
```

Activate it running the following command

On Linux:
```
. .venv/bin/activate
```
On Windows:

```
.venv/Scripts/Activate
```

Alternatively, use
```
conda env create --name <env-name> 
conda activate <env-name>
```

## Install dependencies

**NOTE**: On Windows, there is one more step required. You need to install OpenCV. Either by using `conda install opencv` or following [this tutorial](https://docs.opencv.org/4.x/d3/d52/tutorial_windows_install.html).

Once activated, install all required dependencies via

```
pip install -r requirements.txt
```

# Structure

TODO

# Data

TODO assuming some folder structure

# NOTES

- `setup.py` contains fixed version `2.8.0` because of bug in azure function. See [this](https://stackoverflow.com/questions/73696134/azure-how-to-deploy-a-functions-app-with-tensorflow/73704428#73704428)

# DOCKER

`docker build -t 8710apps.azurecr.io/precipitates/segmentation:latest -f Dockerfile .`

`docker login 8710apps.azurecr.io` and use registry's admin credentials (see Access Keys panel in portal)

`docker push 8710apps.azurecr.io/precipitates/segmentation:latest`

Add secret manually(because it doesn't see `secrets` in portal app)

- `az containerapp update -g rg-8710-department --name ca-8710 --set-env-vars "<connection string>"
