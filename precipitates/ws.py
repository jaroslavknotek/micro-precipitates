import wandb
import precipitates.train
import precipitates.dataset as ds
from datetime import datetime
import pathlib

import argparse
import logging 

import sys
import traceback

logging.basicConfig()
logger = logging.getLogger("prec")
logger.setLevel(logging.DEBUG)



def run_w_data():    

    run = wandb.init(
        #project='my-prec-sweep-small',
        dir="../tmp/",
        entity='knotek',
        save_code = False,
    )
    
    args = wandb.config
    ts = datetime.strftime(datetime.now(),'%Y%m%d%H%M%S')
    model_suffix = '-'.join([ f"{k}={args[k]}" for k in sweep_configuration['parameters']])
    model_path = pathlib.Path(f"../tmp/{ts}_{train_data.parent.name}_{model_suffix}.h5")
       
    try:
        precipitates.train.run_training_w_dataset(
            train_ds,
            val_ds,
            args,
            model_path
        )
    except Exception:
        traceback.print_exc()
        
parser = argparse.ArgumentParser()
parser.add_argument('--crop-size',required =True,type=int)
parser.add_argument('--filter-size',required =False,default=0,type=int)
cli_args = parser.parse_args()

sweep_configuration = {
#    'method': 'random',
    'method':'grid',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_iou'},
    'parameters': 
    {
        'crop_stride':{'values': [32]},
        'patience':{'values': [5]},
        # 'crop_stride':{'values': [256]},
        # 'patience':{'values': [0]},
        #'loss':{'values': ['bc','wbc-1-2','wbc-2-1','wbc-5-1','bfl']}, # 'dwbc', 'wbc-1-2' removed
        'loss':{'values': ['bfl']}, # 'dwbc', 'wbc-1-2' removed
        'filter_size':{'values': [cli_args.filter_size]},
        'cnn_filters':{'values': [8,16,32]},
        'cnn_depth':{'values': [2,3,4]},
        'cnn_activation':{'values': ['elu','relu']},
        'crop_size':{'values': [cli_args.crop_size]}
     }
}

sweep_id = wandb.sweep(
    sweep=sweep_configuration, 
    project='precipitates'
)

    
# Start sweep job.

logging.info("Preparing Dataset")

fixed_config = {
    "crop_stride":32,
    "crop_shape":(cli_args.crop_size,cli_args.crop_size),
    "filter_size":0
}

train_data = "../data/20230427/labeled/"
train_data = pathlib.Path(train_data)
train_ds,val_ds = ds.prepare_datasets(
    train_data,
    crop_stride=fixed_config['crop_stride'],
    crop_shape = fixed_config['crop_shape'],
    filter_size = cli_args.filter_size,
    cache_file_name=f".cache-{fixed_config['crop_shape'][0]}",
    generator=False
)

wandb.agent(sweep_id, function=run_w_data)
wandb.finish()
