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
    test_dir = pathlib.Path("data/test/IN")
    
    assert len(list(test_dir.rglob('*.png')))>0
    try:
        precipitates.train.run_training_w_dataset(
            train_ds,
            val_ds,
            args,
            model_path,
            test_dir
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
        'patience':{'values': [5]},
        'loss':{'values': ['bfl']}, # 'dwbc', 'wbc-1-2' removed
        'filter_size':{'values': [cli_args.filter_size]},
        'cnn_filters':{'values': [16]},
        'cnn_depth':{'values': [8]},
        'cnn_activation':{'values': ['elu']},
        'crop_size':{'values': [cli_args.crop_size]}
     }
}

sweep_id = wandb.sweep(
    sweep=sweep_configuration, 
    project='precipitates-normalized'
)
    
# Start sweep job.

logging.info("Preparing Dataset")

train_data = pathlib.Path("data/20230617-normalized/labeled/")
train_ds,val_ds = ds.prepare_datasets(
    train_data,
    crop_size = cli_args.crop_size,
    filter_size = cli_args.filter_size,
)

wandb.agent(sweep_id, function=run_w_data)
wandb.finish()
