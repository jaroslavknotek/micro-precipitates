import wandb
import precipitates.train
from datetime import datetime
import pathlib

import logging 

import sys
import traceback


logging.basicConfig()
logger = logging.getLogger("prec")
logger.setLevel(logging.DEBUG)

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
        'loss':{'values': ['dwbc','bc','wbc-1-2','wbc-2-1']},
        'filter_size':{'values': [0, 5, 9, 13]},
        'cnn_filters':{'values': [8,16,32]},
        'cnn_depth':{'values': [4,5,6]},
        'cnn_activation':{'values': ['elu','relu']},
        'crop_size':{'values': [64,128]}
     }
}

sweep_id = wandb.sweep(
    sweep=sweep_configuration, 
    project='my-prec-sweep-small'
)

def run():    
    train_data = "../data/20230427/labeled/"
    train_data = pathlib.Path(train_data)
    
    #train_data = pathlib.Path(args.train_data)
    
    run = wandb.init(
        project='my-prec-sweep-small',
        dir="../tmp/wandb_dir",
        entity='knotek',
        save_code = False,
    )
    
    args = wandb.config
    
    ts = datetime.strftime(datetime.now(),'%Y%m%d%H%M%S')
    model_suffix = '_'.join([ f"{k}-{args[k]}" for k in sweep_configuration['parameters']])
    model_path = pathlib.Path(f"../tmp/{ts}_{train_data.parent.name}_{model_suffix}.h5")
    
    #ts = datetime.strftime(datetime.now(),"%Y%m%d%H%M%S")
    #output_dir= pathlib.Path(f"../tmp/{ts}")
    #output_dir= pathlib.Path(f"../tmp")
    #output_dir= pathlib.Path(args.output)
    #output_dir.mkdir(exist_ok=True,parents=True)
    #logger.debug(f"output: {output_dir}")
    
#     with open(output_dir/"params.txt",'w') as f:
#         json.dump(args.__dict__, f, indent=4)
#         logging.debug(f"Args: {args.__dict__}")    
    try:
        precipitates.train.run_training(
            train_data,
            args,
            model_path
        )
    except Exception:
        traceback.print_exc()
    
# Start sweep job.
wandb.agent(sweep_id, function=run)
wandb.finish()
