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
    'method': 'random',
    'name': 'sweep',
    'metric': {'goal': 'maximize', 'name': 'val_acc'},
    'parameters': 
    {
        'crop_stride':{'values': [256]},
        'patience':{'values': [1]},
        'loss':{'values': ['dwbc','bc','wbc-1-2','wbc-2-1']},
        #'filter_size':{'values': [5,7,9,11,13]},
        'filter_size':{'values': [9]},
        'cnn_filters':{'values': [8,16,32]},
        'cnn_depth':{'values': [4,5,6]},
        'cnn_activation':{'values': ['elu','relu']}
     }
}

# Initialize sweep by passing in config. 
# (Optional) Provide a name of the project.
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
        dir="/tmp",
        entity='knotek',
        save_code = False,
    )
    
    args = wandb.config
    #ts = datetime.strftime(datetime.now(),"%Y%m%d%H%M%S")
    #output_dir= pathlib.Path(f"../tmp/{ts}")
    output_dir= pathlib.Path(f"../tmp")
    #output_dir= pathlib.Path(args.output)
    output_dir.mkdir(exist_ok=True,parents=True)
    logger.debug(f"output: {output_dir}")
    
#     with open(output_dir/"params.txt",'w') as f:
#         json.dump(args.__dict__, f, indent=4)
#         logging.debug(f"Args: {args.__dict__}")    
    try:
        precipitates.train.run_training(
            train_data,
            args,
            output_dir
        )
    except Exception:
        traceback.print_exc()
    

# def main():
#     run = wandb.init()

#     TODO train data
#     # note that we define values from `wandb.config`  
#     # instead of defining hard values
#     lr  =  wandb.config.lr
#     bs = wandb.config.batch_size
#     epochs = wandb.config.epochs

#     for epoch in np.arange(1, epochs):
#         train_acc, train_loss = train_one_epoch(epoch, lr, bs)
#         val_acc, val_loss = evaluate_one_epoch(epoch)

#         wandb.log({
#             'epoch': epoch, 
#             'train_acc': train_acc,
#             'train_loss': train_loss, 
#             'val_acc': val_acc, 
#             'val_loss': val_loss
#         })

# Start sweep job.
wandb.agent(sweep_id, function=run,count=30)
wandb.finish()