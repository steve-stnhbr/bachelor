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
from lib.vit_keras.vit_keras import vit
import pandas as pd
from datetime import datetime
from net_flops import net_flops
#from keras_flops import get_flops

L_RATIO = .8
TWO_PATHS_SECOND_BLOCK = True
INPUT_SHAPE = (224, 224, 3)
CLASSES = 2

class MetricsToCSVCallback(keras.callbacks.Callback):
    def __init__(self, filepath, frequency, save_freq=100, verbose=False):
        super().__init__()
        self.filepath = filepath
        self.frequency = frequency
        self.save_freq = save_freq
        self.data = []
        self.verbose = verbose
        
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.steps = self.params['steps']
        
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        step = self.current_epoch * self.steps + batch
        
        if batch % self.frequency != 0:
            return
        
        # Capture all available metrics
        metrics = {name: logs.get(name, 0) for name in self.model.metrics_names}
        metrics.update({'epoch': self.current_epoch, 'step': step})
        metrics.update(self.model.get_metrics_result())
        
        self.data.append(metrics)
        
        # Save to CSV periodically
        if step % self.save_freq == 0:
            self.save_to_csv()

    def on_epoch_end(self, epoch, logs=None):
        self.save_to_csv()

    def save_to_csv(self):
        df = pd.DataFrame(self.data)
        df.to_csv(self.filepath, index=False)
        if self.verbose:
            print(f"Metrics saved to {self.filepath}")

    def on_train_end(self, logs=None):
        self.save_to_csv()

def load_transform(paths):
    return cai.datasets.load_images_from_files(paths, target_size=INPUT_SHAPE[:2], lab=True, rescale=True, smart_resize=True)

def get_latest_checkpoint(dir):
    if not os.path.isdir(dir):
        return None
    ckpts = [file for file in os.listdir(dir) if file.endswith("keras")]
    ckpts.sort()
    return os.path.join(dir, ckpts[-1])

def execute(model, name=None, lab=False, batch_size=32, workers=16, resume=False, epochs=25, data_root="_data/combined"):
    if name is None:
        name = type(model).__name__
    print(f"Starting training for {name}")
    if resume:
        print(f"Loading model file from checkpoints/{name}")
        #model_file = tf.train.latest_checkpoint(f"checkpoints/{name}")
        model_file = get_latest_checkpoint(f"checkpoints/{name}")
        if model_file is not None:
            print(f"Loading model from file {model_file}")
            model = keras.models.load_model(model_file)

    
    print("Creating datagen")
    

    TRAIN_DATA_PATH = os.path.join(data_root, "train")
    VAL_DATA_PATH = os.path.join(data_root, "val")
    TEST_DATA_PATH = os.path.join(data_root, "test")

    train_datagen = gen_dataset(TRAIN_DATA_PATH, batch_size=batch_size, lab=lab, input_shape=INPUT_SHAPE)
    val_datagen = gen_dataset(VAL_DATA_PATH, batch_size=batch_size, lab=lab, input_shape=INPUT_SHAPE)
    test_datagen = gen_dataset(TEST_DATA_PATH, batch_size=batch_size, lab=lab, input_shape=INPUT_SHAPE)

    test = train_datagen.take(1).as_numpy_iterator()
    for el in test:
        print(el[0].shape, el[1].shape)
        
    # Calculate total number of samples dynamically
    num_samples = train_datagen.cardinality().numpy() * batch_size

    # Compute total training steps
    total_steps = (num_samples // batch_size) * epochs

    
    scheduler = keras.optimizers.schedules.CosineDecay(
        0.001,
        total_steps,
        alpha=1e-9,
        name="CosineDecay",
        warmup_target=0.01,
        warmup_steps=num_samples // (batch_size * 4),
    )
    
    def get_lr_metric(optimizer):
        def lr(y_true, y_pred):
            return optimizer.learning_rate # I use ._decayed_lr method instead of .lr
        return lr

    opt = keras.optimizers.AdamW(learning_rate=scheduler)
    lr_metric = get_lr_metric(opt)
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=opt,
        metrics=[
            keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC(),
            lr_metric
        ],
    )
        
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, min_delta=.01),
        keras.callbacks.ModelCheckpoint(filepath='checkpoints/##name##/model##name##.{epoch:02d}.keras'.replace("##name##", name)),
        keras.callbacks.TensorBoard(log_dir=f'./logs/{name}'),
        keras.callbacks.ModelCheckpoint(filepath='out/best##name##.keras'.replace('##name##', name), save_best_only=True, mode='max'),
        MetricsToCSVCallback(f"metrics/{name}.csv", 100),
    ]
    print(f"Beginning training of model {name}")

    history = model.fit(train_datagen, epochs=epochs, callbacks=callbacks, validation_data=val_datagen)
    pd.DataFrame(history.history).to_csv(f"metrics/final_{name}.csv")

    print("Training finished, starting test evaluation")

    result = model.evaluate(test_datagen)
    print(result)

def gen_dataset(path, batch_size, lab, input_shape, aug=True, deterministic=False):
    class ConditionalRandomBrightness(keras.layers.Layer):
        def __init__(self, factor=0.2, **kwargs):
            super().__init__(**kwargs)
            self.random_brightness = keras.layers.RandomBrightness(factor)

        def call(self, inputs):
            # Create a mask for non-black pixels (any channel > 0)
            non_black_mask = tf.reduce_any(inputs > 0, axis=-1, keepdims=True)

            # Apply the brightness transformation
            transformed = self.random_brightness(inputs)

            # Combine original and transformed pixels
            outputs = tf.where(non_black_mask, transformed, inputs)
            return outputs
        
        def compute_output_shape(self, input_shape):
            # Output shape is the same as the input shape
            return input_shape

    class ConditionalRandomContrast(keras.layers.Layer):
        def __init__(self, factor=0.2, **kwargs):
            super().__init__(**kwargs)
            self.random_contrast = keras.layers.RandomContrast(factor)

        def call(self, inputs):
            # Create a mask for non-black pixels (any channel > 0)
            non_black_mask = tf.reduce_any(inputs > 0, axis=-1, keepdims=True)

            # Apply the brightness transformation
            transformed = self.random_contrast(inputs)

            # Combine original and transformed pixels
            outputs = tf.where(non_black_mask, transformed, inputs)
            return outputs
        
        def compute_output_shape(self, input_shape):
            # Output shape is the same as the input shape
            return input_shape
    
    data_augmentation = tf.keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(.8, fill_mode="constant"),
        #ConditionalRandomBrightness(.4),
        #ConditionalRandomContrast(.4),
        keras.layers.RandomZoom((-.2, .2), (-.2, .2), fill_mode="constant"),
        keras.layers.Resizing(INPUT_SHAPE[0], INPUT_SHAPE[1])
    ])
    @tf.function()
    def augment(image, label):
        if (aug):
            return data_augmentation(image), to_categorical(label, num_classes=CLASSES)
        else:
            return image, to_categorical(label, num_classes=CLASSES)
    datagen = keras.utils.image_dataset_from_directory(path, batch_size=batch_size, image_size=input_shape[:2], crop_to_aspect_ratio=True, labels="inferred", label_mode="binary")
    if lab:
        datagen = datagen.map(
            lambda x, y: (transform_wrapper(x, target_size=INPUT_SHAPE[:2], rescale=True, smart_resize=True, lab=True), y)
        )
    datagen = datagen\
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE, deterministic=deterministic)\
        .prefetch(tf.data.AUTOTUNE)
    return datagen

@click.command()
@click.option("-w", "--workers", type=int)
@click.option("-b", "--batch_size", type=int)
@click.option('-r', '--resume', is_flag=True)
@click.option("-s", "--start", type=int)
@click.option('-e', '--epochs', type=int)
@click.option('-f', '--print_flops', is_flag=True)
@click.option('-d', '--data_root', type=str, default="_data/combined")
def main(workers, batch_size, resume, start, epochs, print_flops, data_root):
    models = [
        (
            vit.vit_l32(
                image_size=INPUT_SHAPE[:2],
                activation='sigmoid',
                pretrained=False,
                include_top=True,
                pretrained_top=False,
                classes=CLASSES
            ),
            "VisionTransformer"
        ),
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
            vgg19_model(
                img_shape=INPUT_SHAPE, 
                n_classes=CLASSES, 
                weights=None
            ),
            "VGG19"
        ),
    ]
    if print_flops:
        for model, name in models:
            print(f"Name: {name}")
            net_flops(model)
    if start is not None:
        models = models[start:]
    for lab in [False]:
        for model, name in models:
            execute(model, 
                    f"{name}_{'lab' if lab else 'rgb'}", 
                    lab, 
                    workers=workers, 
                    batch_size=batch_size, 
                    resume=resume, 
                    epochs=epochs,
                    data_root=data_root)


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