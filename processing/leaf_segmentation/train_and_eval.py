import cai.datasets
import cai
import os
from tensorflow import keras
import keras.applications
import click
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
from functools import partial
import skimage.color as skimage_color
import cv2
import keras_cv
from losses import DiceLoss
import lib.Mask_RCNN.mrcnn.config as mrcnn_config
import lib.Mask_RCNN.mrcnn.model as modellib
from data import CustomMRCNNDataset

INPUT_SHAPE = (224, 224, 3)
CLASSES = 25

TRAIN_DATA_PATH = os.path.join("_data", "PhenoBench", "train")
VAL_DATA_PATH = os.path.join("_data", "PhenoBench", "val")
TEST_DATA_PATH = os.path.join("_data", "PhenoBench", "test")

MASK_SUBDIR = "leaf_instances"

def load_transform(paths):
    return cai.datasets.load_images_from_files(paths, target_size=INPUT_SHAPE[:2], lab=True, rescale=True, smart_resize=True)

def execute(model, name=None, lab=False, batch_size=32, epochs=15, data='_data', train_data=None, val_data=None):
    if name is None:
        name = type(model).__name__
    print(f"Starting training for {name}")

    opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(
        loss=DiceLoss(),
        optimizer=opt,
        metrics=[
            keras.metrics.OneHotMeanIoU(
                num_classes=CLASSES, sparse_y_pred=False
            ),
            keras.metrics.CategoricalAccuracy(),
        ],
    )

    print("Creating datagen")

    if train_data is None:
        train_dir = os.path.join(data, 'train')
        train_data = gen_dataset(train_dir, MASK_SUBDIR, batch_size=batch_size, lab=lab)
    if val_data is None:
        val_dir = os.path.join(data, 'val')
        val_data = gen_dataset(val_dir, MASK_SUBDIR, batch_size=batch_size, lab=lab)
    # test_dir = os.path.join(data, 'test')
    #test_datagen = gen_dataset(TEST_DATA_PATH, MASK_SUBDIR, batch_size=batch_size, lab=lab)

    callbacks = [
        #keras.callbacks.EarlyStopping(patience=5),
        keras.callbacks.ModelCheckpoint(filepath='checkpoints/model##name##.{epoch:02d}.keras'.replace("##name##", name)),
        keras.callbacks.TensorBoard(log_dir='./logs'),
        keras.callbacks.ModelCheckpoint(filepath='out/best##name##.keras'.replace('##name##', name), save_best_only=True, mode='max', monitor='val_accuracy')
    ]

    model.summary()

    print(f"Beginning training of model {name}")

    model.fit(train_data, epochs=epochs, callbacks=callbacks, validation_data=val_data)

    print("Training finished, starting test evaluation")

    result = model.evaluate(val_data)
    print(result)

def gen_dataset(path, mask_subdir, batch_size, lab):
    x = keras.utils.image_dataset_from_directory(os.path.join(path, "images"), 
                                                 batch_size=batch_size, 
                                                 image_size=INPUT_SHAPE[:2], 
                                                 crop_to_aspect_ratio=True, 
                                                 labels=None).map(lambda x0: x0 / 255).map(lambda x1: tf.expand_dims(x1, 0) if len(x1.shape) == 3 else x1)
    y = keras.utils.image_dataset_from_directory(os.path.join(path, mask_subdir), 
                                                 batch_size=batch_size,
                                                 image_size=INPUT_SHAPE[:2], 
                                                 crop_to_aspect_ratio=True, 
                                                 labels=None,
                                                 color_mode='grayscale').map(lambda y: tf.expand_dims(to_categorical(y, num_classes=CLASSES), 0))
    datagen = tf.data.Dataset.zip((x, y))
    if lab:
        datagen = datagen.map(
            lambda x, y: (transform_wrapper(x, target_size=INPUT_SHAPE[:2], rescale=True, smart_resize=True, lab=True), y)
        , num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    datagen = datagen.prefetch(tf.data.AUTOTUNE)
    for s in datagen.take(5).as_numpy_iterator():
        print(s[0].shape, tf.reduce_max(s[0]).numpy())
        print(s[1].shape, tf.reduce_max(s[1]).numpy())
    return datagen

@click.command()
@click.option("-b", "--batch_size", type=int, default=20)
@click.option("-e", "--epochs", type=int)
@click.option('-d', '--data', type=str)
def main(batch_size, epochs, data):
    class InferenceConfig(mrcnn_config.Config):
        NAME="mask_rcnn"
        NUM_CLASSES = CLASSES # COCO has 80 classes
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

        def compute_backbone_shapes(self, image_shape):
            """Computes the width and height of each stage of the backbone network."""
            return np.array(
                [[int(np.ceil(image_shape[0] / stride)),
                int(np.ceil(image_shape[1] / stride))]
                for stride in self.BACKBONE_STRIDES]
            )

        def __init__(self):
            super().__init__()
            self.BACKBONE_STRIDES = [4, 8, 16, 32, 64]
            self.BACKBONE_SHAPES = self.compute_backbone_shapes([self.IMAGE_MIN_DIM, self.IMAGE_MAX_DIM])

    mrcnn_config_instance = InferenceConfig()
    mrcnn_train_data = CustomMRCNNDataset(os.path.join(data, "train", "images"), os.path.join(data, "train", MASK_SUBDIR), batch_size=batch_size, image_size=INPUT_SHAPE[:2], config=mrcnn_config_instance)
    mrcnn_val_data = CustomMRCNNDataset(os.path.join(data, "val", "images"), os.path.join(data, "val", MASK_SUBDIR), batch_size=batch_size, image_size=INPUT_SHAPE[:2], config=mrcnn_config_instance)
    models = [
        (
            modellib.MaskRCNN(mode="inference", model_dir=os.getcwd(), config=mrcnn_config_instance).keras_model,
            "Mask R-CNN",
            mrcnn_train_data,
            mrcnn_val_data
        ),
        (
            keras_cv.models.DeepLabV3Plus.from_preset("resnet152", num_classes=CLASSES),
            "DeepLabV3Plus_resnet152",
            None,
            None
        ),
        # (
        #     seg_net(INPUT_SHAPE, CLASSES),
        #     "SegNet"
        # )
    ]

    for lab in [False]:
        for model, name, train, val in models:
            execute(model, 
                    f"{name}_{'lab' if lab else 'rgb'}", 
                    lab, 
                    batch_size=batch_size, 
                    epochs=epochs, 
                    data=data, 
                    train_data=train,
                    val_data=val)


def transform(imgs, target_size=(224,224), smart_resize=False, lab=False, rescale=False, bipolar=False):
    arr = []
    for i in range(imgs.shape[0]):
        img = imgs[i]
        img = img.numpy().astype(np.float32)
        def local_rescale(img,  lab):
            if (lab):
                # JP prefers bipolar input [-2,+2]
                if (bipolar):
                    img[:,:,0:3] /= [25, 50, 50]
                    img[:,:,0] -= 2.
                else:
                    img[:,:,0:3] /= [100, 200, 200]
                    img[:,:,1:3] += 0.5
            else:
                if (bipolar):
                    img /= 64.
                    img -= 2.
                else:
                    img /= 255.
        def add_padding_to_make_img_array_squared(img):
            """ Adds padding to make the image squared.
            # Arguments
                img: an image as an array.
            """
            sizex = img.shape[0]
            sizey = img.shape[1]
            if (sizex == sizey):
                return img
            else:
                maxsize = np.max([sizex, sizey])
                padx = (maxsize - sizex) // 2
                pady = (maxsize - sizey) // 2
                return np.pad(img, pad_width=((padx, maxsize - sizex - padx), (pady, maxsize - sizey - pady), (0, 0)))
                #return tf.pad(img, [[padx, padx], [pady, pady]])
        
        def pad_to_square(img):
            # Get the current size of the image
            shape = tf.shape(img)
            height, width = shape[0], shape[1]

            # Determine the size of the new square image
            max_dim = tf.maximum(height, width)

            # Pad the image to make it square
            squared_img = tf.image.resize_with_crop_or_pad(img, target_height=max_dim, target_width=max_dim)

            return squared_img.numpy()
        if (smart_resize):
            if (lab):
                img /= 255
                img = skimage_color.rgb2lab(img)
            if(rescale):
                local_rescale(img,  lab)
            img = pad_to_square(img)
            if ((img.shape[0] != target_size[0]) or (img.shape[1] != target_size[1])):
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)
        else:
            if (lab):
                img /= 255.
                img = skimage_color.rgb2lab(img)
            if(rescale):
                local_rescale(img,  lab)
        arr.append(img)
    return tf.convert_to_tensor(np.stack(arr), dtype=tf.float32)

def transform_wrapper(imgs, target_size=(224,224), smart_resize=False, lab=False, rescale=False, bipolar=False):
    size = imgs.shape[0]
    p = partial(transform, target_size=target_size, smart_resize=smart_resize, lab=lab, rescale=rescale, bipolar=bipolar)
    imgs = tf.py_function(func=p, inp=[imgs], Tout=tf.float32)
    imgs.set_shape((size,) + target_size + (3,))
    return imgs

if __name__ == '__main__':
    main()