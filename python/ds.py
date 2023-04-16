import dataset
import tensorflow as tf
import pathlib

import nn
from keras.callbacks import EarlyStopping, ModelCheckpoint

dump_output =pathlib.Path("../tmp/20230415")
if not dump_output.exists():
    dataset.dump_dataset(
        "../data/20230415/labeled",
        dump_output)

model = nn.compose_unet(128,128,3)
mask_path = pathlib.Path(dump_output/"mask")
img_path = pathlib.Path(dump_output/"img")
model_path = pathlib.Path(dump_output/'model.h5')

batch_size = 32
image_size = 128
val_split = .2

earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint(model_path, verbose=1, save_best_only=True)

# df_X = tf.keras.utils.image_dataset_from_directory(
#     img_path,
#     seed=123,
#     labels=None,
#     image_size=(image_size, image_size),
# #    validation_split = val_split,
#     batch_size=batch_size)
# 
# df_y = tf.keras.utils.image_dataset_from_directory(
#     mask_path,
#     seed=123,
#     labels=None,
#     color_mode='grayscale',
#     image_size=(image_size, image_size),
# #    validation_split = val_split,
#     batch_size=batch_size
# ).map(lambda x:x/255)
# 
# ds = tf.data.Dataset.zip((df_X,df_y))

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

train_ds = tf.data.Dataset.zip((train_X,train_y))


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

val_ds = tf.data.Dataset.zip((val_X,val_y))

results = model.fit(
    train_ds,
    validation_data= val_ds,
    epochs=50,
    callbacks=[earlystopper,checkpointer]
)


