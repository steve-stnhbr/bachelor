import cai.datasets
from schuler_two_branch.test_datagen import PlantLeafsDataGenBinary
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
from models import alexnet_model, vgg19_model, lenet_model

L_RATIO = .8
TWO_PATHS_SECOND_BLOCK = True
INPUT_SHAPE = (224, 224, 3)
CLASSES = 2

TRAIN_DATA_PATH = os.path.join("_data", "train_b")
VAL_DATA_PATH = os.path.join("_data", "val_b")
TEST_DATA_PATH = os.path.join("_data", "test_b")

def load_transform(paths):
    return cai.datasets.load_images_from_files(paths, target_size=INPUT_SHAPE[:2], lab=True, rescale=True, smart_resize=True)

def execute(model, name=None, lab=False, batch_size=32, workers=16):
    if name is None:
        name = type(model).__name__
    print(f"Starting training for {name}")

    opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    
    print("Creating datagen")
    # train_datagen = PlantLeafsDataGenBinary(TRAIN_DATA_PATH, transforms=[load_transform] if lab else None, batch_size=batch_size, workers=workers, use_multiprocessing=True)
    # val_datagen = PlantLeafsDataGenBinary(VAL_DATA_PATH, transforms=[load_transform] if lab else None, batch_size=batch_size, workers=workers, use_multiprocessing=True)
    # test_datagen = PlantLeafsDataGenBinary(TEST_DATA_PATH, transforms=[load_transform] if lab else None, batch_size=batch_size, workers=workers, use_multiprocessing=True)

    train_datagen = gen_dataset(TRAIN_DATA_PATH, batch_size=batch_size, lab=lab)
    val_datagen = gen_dataset(VAL_DATA_PATH, batch_size=batch_size, lab=lab)
    test_datagen = gen_dataset(TEST_DATA_PATH, batch_size=batch_size, lab=lab)

    test = train_datagen.take(5).as_numpy_iterator()
    for el in test:
        print(el[0].shape, el[1].shape)

    print("Dataset sizes [train, val, test]", len(train_datagen), len(val_datagen), len(test_datagen))

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5),
        keras.callbacks.ModelCheckpoint(filepath='checkpoints/model##name##.{epoch:02d}.keras'.replace("##name##", name)),
        keras.callbacks.TensorBoard(log_dir='./logs'),
        keras.callbacks.ModelCheckpoint(filepath='out/best##name##.keras'.replace('##name##', name), save_best_only=True, mode='max', monitor='val_accuracy')
    ]
    print(f"Beginning training of model {name}")

    model.fit(train_datagen, epochs=15, callbacks=callbacks, validation_data=val_datagen)

    print("Training finished, starting test evaluation")

    result = model.evaluate(test_datagen)
    print(result)

def gen_dataset(path, batch_size, lab):
    def map_data(x, y):
        return (x, to_categorical(y, num_classes=2))
    datagen = keras.utils.image_dataset_from_directory(path, batch_size=batch_size, image_size=INPUT_SHAPE[:2], crop_to_aspect_ratio=True, labels="inferred", label_mode="binary")
    if lab:
        datagen = datagen.map(
            lambda x, y: (transform_wrapper(x, target_size=INPUT_SHAPE[:2], rescale=True, smart_resize=True, lab=True), y)
        )
    datagen = datagen.map(map_data, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False).prefetch(tf.data.AUTOTUNE)
    return datagen

@click.command()
@click.option("-w", "--workers", type=int)
@click.option("-b", "--batch_size", type=int)
def main(workers, batch_size):
    models = [
        (
            keras.applications.ResNet152V2(
                include_top = True,
                input_shape=INPUT_SHAPE,
                classes=CLASSES,
                weights=None,
                pooling='max',
            ),
            "ResNet152V2"
        ),
        (
            keras.applications.InceptionV3(
                include_top=True,
                input_shape=INPUT_SHAPE,
                classes=CLASSES,
                weights=None,
                pooling='max',
            ),
            "InceptionV3"
        ),
        (
            keras.applications.MobileNetV3Large(
                include_top=True,
                input_shape=INPUT_SHAPE,
                classes=CLASSES,
                weights=None,
                pooling='max',
            ),
            "MobileNetV3Large"
        ),
        (
            keras.applications.ConvNeXtLarge(
                include_top=True,
                input_shape=INPUT_SHAPE,
                classes=CLASSES,
                weights=None,
                pooling='max',
            ),
            "ConvNeXtLarge"
        ),
        (
            alexnet_model(
                img_shape=INPUT_SHAPE, 
                n_classes=CLASSES, 
                weights=None
            ),
            "AlexNet"
        ),
        (
            lenet_model(
                img_shape=INPUT_SHAPE, 
                n_classes=CLASSES, 
                weights=None
            ),
            "LeNet"
        ),
        (
            vgg19_model(
                img_shape=INPUT_SHAPE, 
                n_classes=CLASSES, 
                weights=None
            ),
            "VGG19"
        )
    ]

    for lab in [True]:
        for model, name in models:
            execute(model, f"{name}_{'lab' if lab else 'rgb'}", lab, workers=workers, batch_size=batch_size)


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