import wandb
import precipitates.dataset as ds
import precipitates.training as training
        
import precipitates.evaluation as evaluation
import precipitates.visualization as visualization

import precipitates.nnet as nnet
from datetime import datetime
import pathlib

import argparse
import logging 

import sys
import traceback

import functools
import numpy as np

import torch
from torch import nn

logging.basicConfig()
logger = logging.getLogger("prec")
logger.setLevel(logging.DEBUG)

def run_sweep(
    dataset_array, 
    named_test_data,
    results_dir_root,
    patience = 5,
    device_name='cuda',
    repeat = 100
):
    _ = wandb.init(
        dir="wandb-tmp/",
        entity='knotek',
        save_code = False,
    )
    
    args = wandb.config
    apply_weight_map = args.apply_weight_map == 1
    
    if np.log2(args.crop_size) < args.cnn_depth +2:
        logger.warn(f"Cannot have crop_size={args.crop_size} and cnn_depth={args.cnn_depth}")
        return
    
    try:    
        device = torch.device(device_name)
        
        ts = datetime.strftime(datetime.now(),'%Y%m%d%H%M%S')
        model_suffix = '-'.join([ f"{k}={args[k]}" for k in sweep_configuration['parameters'] if not k.startswith('_') ])
        model_eval_root = results_dir_root/f"{ts}-{model_suffix}"
        
        
        # exclude only_denoise images
        if args.loss_denoise_weight == 0:
            dataset_array = [ (img,mask) for img,mask in dataset_array if mask is not None]
        
        # train
                
        train_dataloader,val_dataloader = ds.prepare_train_val_dataset(
            dataset_array,
            args.crop_size,
            apply_weight_map,
            repeat = repeat
        )

        model = nnet.UNet(
            start_filters=args.cnn_filters, 
            depth=args.cnn_depth, 
            in_channels=3,
            out_channels=4
            #up_mode='bicubic'
        )

        loss = nnet.resolve_loss(args.loss)
        loss_denoise = nn.MSELoss().to(device)
        optimizer = torch.optim.Adam(model.parameters())
        model.to(device)
    
        wandb.watch(model,log='parameters')
        
        model,loss_dict = training.train_model(
            model,
            train_dataloader,
            val_dataloader,
            optimizer,
            loss_denoise,
            loss,
            wandb=wandb,
            denoise_loss_weight = args.loss_denoise_weight,
            device=device,
            patience = patience
        )
        
        model_eval_root.mkdir(exist_ok=True,parents=True)
        model_tmp_path =model_eval_root/ f'model.torch'
        
        wandb.unwatch(model)
        torch.save(model, model_tmp_path)
        
        test_data,test_names = named_test_data
        evaluations = evaluation.evaluate_model(model, test_data,test_names,args.crop_size)
        visualization.save_evaluations(model_eval_root, evaluations,loss_dict)
        
        
        aggregated = {}
        metrics_list = [v['metrics'][-1] for k,v in evaluations.items()]
        for k in metrics_list[0]:
            aggregated[k] = np.mean([d[k] for d in metrics_list])    
            aggregated[k] = np.mean([d[k] for d in metrics_list])
        wandb.log(aggregated)
        
        
        logger.info('Saved evaluation visualization')
    except Exception as e:
        logging.error("e", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    cli_args = parser.parse_args()

    sweep_configuration = {
        'method': 'random',
        'name': 'sweep',
        'metric': {'goal': 'minimize', 'name': 'val_loss'},
        'parameters': 
        {
            # 'crop_size':{'values':[64,128,256,512]},
            # 'cnn_depth':{'values':[3,4,5,6,7,8]},
            'apply_weight_map': {'values':[0,1]},
            'crop_size':{'values':[256]},
            'cnn_depth':{'values':[6]},
            'loss':{'values': ['fl']},
            'loss_denoise_weight':{'values':[1,10,50]},
            'cnn_filters':{'values': [8,16]},
        }
    }

    sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project='precipitates-20230728'
    )

    logging.info("Preparing Dataset")
    
    dataset_root = pathlib.Path('data/20230724/labeled/')
    denoise_root = pathlib.Path('../delisa-new-up/')
    test_data_path = pathlib.Path('data/test/')
    assert len(list(dataset_root.rglob("*.png")))>0
    assert len(list(denoise_root.rglob("*.tif")))>0
    assert len(list(test_data_path.rglob("*.png")))>0
    
    results_dir_root = pathlib.Path('tmp')
    
    named_test_data= ds.load_img_mask_pair(test_data_path,append_names=True)
    dataset_array = ds.load_with_denoise(dataset_root,denoise_root)
    logging.info(f"Dataset size: {len(dataset_array)}")
    
    fn =functools.partial(run_sweep,dataset_array,named_test_data,results_dir_root)
    wandb.agent(sweep_id, function=fn)
    wandb.finish()
