import dataset
import tensorflow as tf
import pathlib
import imageio
import nn
from keras.callbacks import EarlyStopping, ModelCheckpoint
import precipitates

dump_output =pathlib.Path("../tmp/20230417-morp_opening")
if not dump_output.exists():
    dataset.dump_dataset(
        "../data/20230417-morp_opening/labeled",
        dump_output)

model = nn.compose_unet(128,128,3)
mask_path = pathlib.Path(dump_output/"mask")
img_path = pathlib.Path(dump_output/"img")
model_path = pathlib.Path(dump_output/'model-20230417-morp_opening.h5')

batch_size = 32
image_size = 128
val_split = .2

earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint(model_path, verbose=1, save_best_only=True)

test_img1 = precipitates.load_microscope_img("../data/test/DELISA LTO_08Ch18N10T_pricny rez_nulty stav_TOP_BSE_09_JR/img.png")
test_img2 = precipitates.load_microscope_img('../data/20230415/not_labeled/DELISA LTO_08Ch18N10T-podelny rez-nulty stav_BSE_01_TiC,N_03_224px10um.tif')

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        for i,img in enumerate([test_img1,test_img2]):
            pred = nn.predict(model,img)
            path = dump_output/f'test_{i}_{epoch:03}.png'
            imageio.imwrite(path, pred)

train_X = tf.keras.utils.image_dataset_from_directory(
    img_path,
    seed=123,
    subset='training',
    labels=None,
    image_size=(image_size, image_size),
    validation_split = val_split,
    batch_size=batch_size)

train_y = tf.keras.utils.image_dataset_from_directory(
    mask_path,
    seed=123,
    labels=None,
    subset='training',
    color_mode='grayscale',
    image_size=(image_size, image_size),
    validation_split = val_split,
    batch_size=batch_size
).map(lambda x:x/255)

train_ds = tf.data.Dataset.zip((train_X,train_y)).prefetch(tf.data.AUTOTUNE)


val_X = tf.keras.utils.image_dataset_from_directory(
    img_path,
    seed=123,
    subset='validation',
    labels=None,
    image_size=(image_size, image_size),
    validation_split = val_split,
    batch_size=batch_size)

val_y = tf.keras.utils.image_dataset_from_directory(
    mask_path,
    seed=123,
    labels=None,
    subset='validation',
    color_mode='grayscale',
    image_size=(image_size, image_size),
    validation_split = val_split,
    batch_size=batch_size
).map(lambda x:x/255)

val_ds = tf.data.Dataset.zip((val_X,val_y)).prefetch(tf.data.AUTOTUNE)

results = model.fit(
    train_ds,
    validation_data= val_ds,
    epochs=50,
    callbacks=[earlystopper,checkpointer,DisplayCallback()]
)

