import precipitates.dataset as ds
import tensorflow as tf
import pathlib
import imageio
import precipitates.nn
from keras.callbacks import EarlyStopping, ModelCheckpoint
import precipitates.precipitate as precipitate
import precipitates.evaluation as evaluation
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
    
    def __init__(self,dump_output,model,test_img_mask_pair,filter_size=0):
        self.dump_output= dump_output
        self.model = model
        self.test_img_mask_pair = test_img_mask_pair
        self.filter_size = filter_size

    def on_epoch_end(self, epoch, logs=None):
        for i,(img,mask) in enumerate(self.test_img_mask_pair):
            path = self.dump_output/f'test_{i}_{epoch:03}.png'
            json_path = self.dump_output/f'test_{i}_{epoch:03}.json'
            
            (img,ground_truth,pred,metrics_res) = evaluation.evaluate(
                self.model,
                img,
                mask,
                self.filter_size
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
            plt.savefig(path)
            #json.dump(metrics_res,open(json_path,'w'))
            logger.info(f"Epoch {epoch} img:{i}: {json.dumps(metrics_res,indent=4)}")
            
            print("LOGS keys", list(logs.keys()))
            logged = {
                'epoch': epoch, 
                'image_id':i,
                # 'train_acc': train_acc,
                # 'train_loss': train_loss, 
                # 'val_acc': val_acc, 
                'val_loss': logs.get("loss"),
            }
            logged.update(metrics_res[-1])
            wandb.log(logged)
            
            
def _norm(img):

    img_min=np.min(img)
    img_max = np.max(img)
    return (img.astype(float)-img_min) / (img_max-img_min)

def run_training(
    train_data,
    args,
    dump_output
):    
    training_timestamp = datetime.strftime(datetime.now(),'%Y%m%d%H%M%S')
    test_dir = pathlib.Path("../data/test/IN")
    
    if dump_output is None:
        dump_output =pathlib.Path("../tmp/")/training_timestamp
        dump_output.mkdir(exist_ok=True,parents=True)
    logging.debug("output:",dump_output)

    CROP_SHAPE= (128,128)

    loss = nn.resolve_loss(args.loss)
    model = nn.build_unet(
        CROP_SHAPE,
        loss=loss
    )
    model_path = pathlib.Path(dump_output/'model.h5')
    
    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint(model_path, verbose=1, save_best_only=True)
    
    test_img_mask_pairs = evaluation._read_test_imgs_mask_pairs(test_dir)
    display = DisplayCallback(dump_output, model, test_img_mask_pairs,args.filter_size)
    callbacks = [earlystopper,checkpointer,display]

    logging.info("Reading Dataset")
    train_ds,val_ds = ds.prepare_datasets(
        train_data,
        crop_stride=args.crop_stride,
        filter_size = args.filter_size
    )
    
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
    parser.add_argument('--crop-stride',required =True,type=int)
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
