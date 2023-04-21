import precipitates.dataset as ds
import tensorflow as tf
import pathlib
import imageio
import nn
from keras.callbacks import EarlyStopping, ModelCheckpoint
import precipitates.precipitate as precipitate

from datetime import datetime


class DisplayCallback(tf.keras.callbacks.Callback):
        def __init__(self,dump_output,model,test_imgs):
            sefl.dump_output= dump_output
            self.model = model
            self.test_imgs=test_imgs

        def on_epoch_end(self, epoch, logs=None):
            for i,img in enumerate(self.test_imgs):
                pred = nn.predict(self.model,img)
                path = self.dump_output/f'test_{i}_{epoch:03}.png'
                imageio.imwrite(path, pred)

                

def run(train_data,dump_output=None,crop_stride = 8):
    
    training_timestamp = datetime.strftime(datetime.now(),'%Y%m%d%H%M%S')
    
    if dump_output is None:
        dump_output =pathlib.Path("../tmp/")/training_timestamp
        dump_output.mkdir(exist_ok=True,parents=True)

    CROP_SHAPE= (128,128)

    test_img1 = precipitate.load_microscope_img("../data/test/DELISA LTO_08Ch18N10T_pricny rez_nulty stav_TOP_BSE_09_JR/img.png")
    test_img2 = precipitate.load_microscope_img('../data/20230415/not_labeled/DELISA LTO_08Ch18N10T-podelny rez-nulty stav_BSE_01_TiC,N_03_224px10um.tif')
    test_imgs = [test_img1,test_img2]

    model = nn.compose_unet(CROP_SHAPE)
    model_path = pathlib.Path(dump_output/'model-20230419.h5')

    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint(model_path, verbose=1, save_best_only=True)
    display = DisplayCallback(dump_output, model, test_imgs)
    callbacks = [earlystopper,checkpointer,display]

    train_ds,val_ds,_ = ds.prepare_datasets(train_data,crop_stride=crop_stride)

    results = model.fit(
        train_ds,
        validation_data= val_ds,
        epochs=50,
        callbacks=callbacks
    )

if __name__ == "__main__":
    dump_output =pathlib.Path("../tmp/20230419")
    train_data = pathlib.Path("../data/20230415/labeled")
    
    run_training(train_data,dump_output)
