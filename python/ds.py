import dataset as ds
import tensorflow as tf
import pathlib
import imageio
import nn
from keras.callbacks import EarlyStopping, ModelCheckpoint
import precipitates


from datetime import datetime

training_timestamp = datetime.strftime(datetime.now(),'%Y%m%d%H%M%S')

print(training_timestamp)


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for i,img in enumerate([test_img1,test_img2]):
            pred = nn.predict(model,img)
            path = dump_output/f'test_{i}_{epoch:03}.png'
            imageio.imwrite(path, pred)

            

CROP_SHAPE= (128,128)
dump_output =pathlib.Path("../tmp/20230419")
train_data = pathlib.Path("../data/20230415/")

model = nn.compose_unet(CROP_SHAPE)
model_path = pathlib.Path(dump_output/'model-20230419.h5')

earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint(model_path, verbose=1, save_best_only=True)
display = DisplayCallback()
callbacks = [earlystopper,checkpointer,display]

test_img1 = precipitates.load_microscope_img("../data/test/DELISA LTO_08Ch18N10T_pricny rez_nulty stav_TOP_BSE_09_JR/img.png")
test_img2 = precipitates.load_microscope_img('../data/20230415/not_labeled/DELISA LTO_08Ch18N10T-podelny rez-nulty stav_BSE_01_TiC,N_03_224px10um.tif')

# train_ds,val_ds,spe = ds.prepare_datasets(train_data,crop_stride=8)
train_ds,val_ds,spe = ds.prepare_datasets(train_data,crop_stride=256)

for _ in range(10):
    
    print('it', len([None for x in train_ds]))

results = model.fit(
    train_ds,
    validation_data= val_ds,
    epochs=50,
    #steps_per_epoch = spe,
    callbacks=callbacks
)
