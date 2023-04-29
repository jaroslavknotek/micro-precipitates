import precipitates.dataset as ds
import tensorflow as tf
import pathlib
import imageio
import nn
from keras.callbacks import EarlyStopping, ModelCheckpoint
import precipitates.precipitate as precipitate
import precipitates.evaluation as evaluation
import sys
import logging
from datetime import datetime
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt

logging.basicConfig()
logger = logging.getLogger("prec")
logger.setLevel(logging.DEBUG)

class DisplayCallback(tf.keras.callbacks.Callback):
    
    def __init__(self,dump_output,model,test_img_mask_pair,filter_small=False):
        self.dump_output= dump_output
        self.model = model
        self.test_img_mask_pair = test_img_mask_pair
        self.filter_small = filter_small

    def on_epoch_end(self, epoch, logs=None):
        for i,(img,mask) in enumerate(self.test_img_mask_pair):
            path = self.dump_output/f'test_{i}_{epoch:03}.png'
            json_path = self.dump_output/f'test_{i}_{epoch:03}.json'
            
            (img,ground_truth,pred,metrics_res) = evaluation.evaluate(
                self.model,
                img,
                mask,
                self.filter_small
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
            json.dump(metrics_res,open(json_path,'w'))
            logger.info(f"Epoch {epoch} img:{i}: {json.dumps(metrics_res,indent=4)}")
            
            

                
def _norm(img):
    img_min=np.min(img)
    img_max = np.max(img)
    return (img.astype(float)-img_min) / (img_max-img_min)

def run_training(
    train_data,
    args,
    dump_output
):
    CROP_SHAPE= (128,128)

    model = nn.compose_unet(
        CROP_SHAPE,
        loss=args.loss,
        weight_zero = args.wbc_weight_zero,
        weight_one = args.wbc_weight_one
    )
    
    model_path = pathlib.Path(dump_output/'model.h5')

    earlystopper = EarlyStopping(patience=args.patience, verbose=1)
    checkpointer = ModelCheckpoint(model_path, verbose=1, save_best_only=True)
    
    callbacks = [earlystopper,checkpointer]
    if args.test_dir is not None:
        test_img_mask_pairs=evaluation._read_test_imgs_mask_pairs(args.test_dir)
        display = DisplayCallback(
            dump_output,
            model, 
            test_img_mask_pairs,
            args.filter_small
        )
        callbacks.append(display)
        
    logger.info("Reading Dataset")
    train_ds,val_ds,spe = ds.prepare_datasets(
        train_data,
        crop_stride=args.crop_stride,
        filter_small=args.filter_small
    )
    logger.info("Started Training")
    logger.debug(f"Expected steps per epoch:{spe}")

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
    
    parser.add_argument('--wbc-weight-zero',required=False,default =1)
    parser.add_argument('--wbc-weight-one',required=False,default=1)

    parser.add_argument('--output',required=False,default=default_output_path)
    parser.add_argument(
        '--filter-small',
        default=True,
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument('--test-dir',required=False)

    return parser.parse_args(args_arr)

if __name__ == "__main__": 
    
    
    args = _parse_args()
    
    
    output_dir= pathlib.Path(args.output)
    output_dir.mkdir(exist_ok=True,parents=True)
    logger.debug("output: {output_dir}")
    
    
    with open(output_dir/"params.txt",'w') as f:
        json.dump(args.__dict__, f, indent=4)
        logging.debug(f"Args: {args.__dict__}")
        

    train_data = pathlib.Path(args.train_data)
    
    
    
    run_training(
        train_data,
        args,
        output_dir
    )
