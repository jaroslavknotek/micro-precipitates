import precipitates.dataset as ds
import tensorflow as tf
import pathlib
import imageio
import precipitates.nn
from keras.callbacks import EarlyStopping, ModelCheckpoint
import precipitates.precipitate as precipitate
import precipitates.evaluation as evaluation
import precipitates.dataset as dataset
import precipitates.nn as nn

import sys
import logging
from datetime import datetime
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt

import wandb


logging.basicConfig()
logger = logging.getLogger("prec")
logger.setLevel(logging.DEBUG)

class DisplayCallback(tf.keras.callbacks.Callback):
    
    def __init__(self,dump_output,model,test_img_mask_pair,args):
        self.dump_output= dump_output
        self.model = model
        self.test_img_mask_pair = test_img_mask_pair
        self.args = args

    def on_epoch_end(self, epoch, logs=None):
        
        metrics_list = []
        for i,(img,mask) in enumerate(self.test_img_mask_pair):
#             path = self.dump_output/f'test_{i}_{epoch:03}.png'
#             json_path = self.dump_output/f'test_{i}_{epoch:03}.json'
            mask = np.uint8(mask)
            (img,ground_truth,pred,metrics_res) = evaluation.evaluate(
                self.model,
                img,
                mask
            )
            
            fig,axs = plt.subplots(1,4,figsize=(16,4))
            evaluation._visualize_pairs(
                axs,
                img,
                mask,
                pred,
                metrics_res,
                "model"
            )
                        
            metrics = {
                'epoch': epoch, 
                'image_id':i,
                'train_loss':logs.get("loss"),
                'train_iou':logs.get("io_u"),
                'val_loss': logs.get("val_loss"),
                'val_iou': logs.get("val_io_u"),
                
            }|metrics_res[-1]
            
            metrics_list.append(metrics)
        
        aggregated = {}
        for k in metrics_list[0]:
            aggregated[k] = np.mean([d[k] for d in metrics_list])
        logger.info(f"Epoch {epoch} img:{i}: {json.dumps(aggregated,indent=4)}")        
        wandb.log(aggregated) 
            
def _norm(img):
    img_min=np.min(img)
    img_max = np.max(img)
    return (img.astype(float)-img_min) / (img_max-img_min)

def run_training(
    train_data,
    args,
    model_path
):   
    logging.info("Reading Dataset")
    train_ds,val_ds = ds.prepare_datasets(
        train_data,
        crop_shape = crop_shape,
        filter_size = args.filter_size
    )
    
    run_training_w_dataset(train_ds,val_ds,args,model_path)
    
    
def run_training_w_dataset(
    train_ds,
    val_ds,
    args,
    model_path, 
    test_dir
):   
    dump_output = None
    
    loss =nn.resolve_loss(args.loss)
    model = nn.build_unet(
        crop_size = args.crop_size,
        loss=loss,
        activation = args.cnn_activation,
        start_filters = args.cnn_filters,
        depth = args.cnn_depth
    )
    
    logger.debug(f"Will checkpiont model to {model_path}")
    #model_path = pathlib.Path(dump_output/model_name)
    
    earlystopper = EarlyStopping(patience=args.patience, verbose=1)
    checkpointer = ModelCheckpoint(model_path, verbose=1, save_best_only=True)
    
    test_img_mask_pairs = dataset.load_img_mask_pair(test_dir,args.filter_size)
    display = DisplayCallback(dump_output, model, test_img_mask_pairs,args)
    callbacks = [earlystopper,checkpointer,display]
    
    results = model.fit(
        train_ds,
        validation_data= val_ds,
        epochs=100,
        callbacks=callbacks
    )
    
    

def _parse_args(args_arr = None):
    training_timestamp = datetime.strftime(datetime.now(),'%Y%m%d%H%M%S')
    
    default_output_path =pathlib.Path("/tmp/")/training_timestamp
    parser = argparse.ArgumentParser()
    parser.add_argument('--patience',default=5,type=int)
    
    parser.add_argument('--loss',default = 'bc',choices = ['dwbc','bc','wbc'])
    parser.add_argument('--train-data',required=True)
    
    parser.add_argument('--output',required=False,default=default_output_path)
    parser.add_argument('--filter-size',default=0,type=int)
    parser.add_argument('--test-dir',required=False)

    return parser.parse_args(args_arr)

if __name__ == "__main__": 
    
    
    args = _parse_args()
    
    
    output_dir= pathlib.Path(args.output)
    output_dir.mkdir(exist_ok=True,parents=True)
    logger.debug(f"output: {output_dir}")
    
    
    with open(output_dir/"params.txt",'w') as f:
        json.dump(args.__dict__, f, indent=4)
        logging.debug(f"Args: {args.__dict__}")
        

    train_data = pathlib.Path(args.train_data)
    
    try:
        run_training(
            train_data,
            args,
            output_dir
        )
    except Exception:
        traceback.print_exc()
