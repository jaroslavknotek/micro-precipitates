import precipitates.dataset as ds
import tensorflow as tf
import pathlib
import imageio
import nn
from keras.callbacks import EarlyStopping, ModelCheckpoint
import precipitates.precipitate as precipitate
import sys
import logging
from datetime import datetime
import argparse
import numpy as np
logging.basicConfig()
logger = logging.getLogger("prec")
logger.setLevel(logging.DEBUG)

class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self,dump_output,model,test_imgs):
        self.dump_output= dump_output
        self.model = model
        self.test_imgs=test_imgs

    def on_epoch_end(self, epoch, logs=None):
        for i,img in enumerate(self.test_imgs):
            pred = nn.predict(self.model,img)
            path = self.dump_output/f'test_{i}_{epoch:03}.png'
            imageio.imwrite(path, pred)

                
def _norm(img):
    img_min=np.min(img)
    img_max = np.max(img)
    return (img.astype(float)-img_min) / (img_max-img_min)

def run_training(
    train_data,
    dump_output=None,
    crop_stride = 8,
    filter_small = False,
    loss='bc',
    test_imgs=[]):

    logger.debug(f"Data: {train_data}")
    logger.debug(f"Data: {dump_output}")
    logger.debug(f"Data: {crop_stride}")
    logger.debug(f"Filter small: {filter_small}")

    CROP_SHAPE= (128,128)

    model = nn.compose_unet(CROP_SHAPE,loss=loss)
    model_path = pathlib.Path(dump_output/'model.h5')

    earlystopper = EarlyStopping(patience=10, verbose=1)
    checkpointer = ModelCheckpoint(model_path, verbose=1, save_best_only=True)
    display = DisplayCallback(dump_output, model, test_imgs)
    callbacks = [earlystopper,checkpointer,display]
    logger.info("Reading Dataset")
    train_ds,val_ds,spe = ds.prepare_datasets(
        train_data,
        crop_stride=crop_stride,
        filter_small=filter_small
    )
    logger.info("Started Training")
    logger.debug(f"Expected steps per epoch:{spe}")

    results = model.fit(
        train_ds,
        validation_data= val_ds,
        epochs=50,
        callbacks=callbacks
    )

def _parse_args(default_output_path, args_arr = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop-stride',required =True,type=int)
    parser.add_argument('--loss',default = 'bc',choices = ['dwbc'])
    parser.add_argument('--train-data',required=True)

    parser.add_argument('--output',required=False,default=def_output)
    parser.add_argument(
        '--filter-small',
        default=True,
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument('--test-imgs',nargs='*',default=[])

    return parser.parse_args(args_arr)

if __name__ == "__main__": 
    
    training_timestamp = datetime.strftime(datetime.now(),'%Y%m%d%H%M%S')

    def_output =pathlib.Path("/tmp/")/training_timestamp
    args = _parse_args(def_output)

    output_dir= pathlib.Path(args.output)
    logger.debug("output:",output_dir)
    output_dir.mkdir(exist_ok=True,parents=True)

    train_data = pathlib.Path(args.train_data)
    
    test_imgs = list(map(precipitate.load_microscope_img, args.test_imgs))
    # HACK
    if test_imgs == []:
        test_img1 = precipitate.load_microscope_img("../data/test/DELISA LTO_08Ch18N10T_pricny rez_nulty stav_TOP_BSE_09_JR/img.png")
        test_img2 = precipitate.load_microscope_img('../data/20230415/not_labeled/DELISA LTO_08Ch18N10T-podelny rez-nulty stav_BSE_01_TiC,N_03_224px10um.tif')
        test_imgs = [(img) for img in [test_img1,test_img2]]

    run_training(
        train_data,
        dump_output = output_dir,
        crop_stride=args.crop_stride,
        filter_small=args.filter_small,
        loss=args.loss,
        test_imgs = test_imgs
    )
