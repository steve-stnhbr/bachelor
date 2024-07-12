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
import keras
from tensorflow.keras import backend as K
from typing import Union, Callable

#from models import build_pspnet
from keras_segmentation.models.pspnet import pspnet_101
from keras_segmentation.models.segnet import vgg_segnet
from keras_segmentation.models.fcn import fcn_32_mobilenet

INPUT_SHAPE = (512, 512, 3)
CLASSES = 25

TRAIN_DATA_PATH = os.path.join("_data", "PhenoBench", "train")
VAL_DATA_PATH = os.path.join("_data", "PhenoBench", "val")
TEST_DATA_PATH = os.path.join("_data", "PhenoBench", "test")

MASK_SUBDIR = "leaf_instances"

def load_transform(paths):
    return cai.datasets.load_images_from_files(paths, target_size=INPUT_SHAPE[:2], lab=True, rescale=True, smart_resize=True)

def execute(model, name=None, lab=False, batch_size=32, epochs=15, data='_data', train_data=None, val_data=None, input_shape=INPUT_SHAPE):
    if name is None:
        name = type(model).__name__
    print(f"Starting training for {name}")

    def iou_loss(y_true, y_pred, smooth=1e-6):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        
        intersection = K.sum(y_true * y_pred)
        total = K.sum(y_true) + K.sum(y_pred)
        union = total - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        return 1 - iou
    
    def multiclass_iou_loss(y_true, y_pred, smooth=1e-6):
        """
        Compute the IoU loss for multiclass segmentation.

        :param y_true: True labels, one-hot encoded, shape (batch_size, height, width, num_classes)
        :param y_pred: Predictions, shape (batch_size, height, width, num_classes)
        :param smooth: Smoothing factor to avoid division by zero
        :return: Average IoU loss across all classes
        """
        num_classes = y_pred.shape[-1]
        iou_loss_per_class = []

        for c in range(num_classes):
            y_true_c = y_true[..., c]
            y_pred_c = y_pred[..., c]
            
            intersection = K.sum(y_true_c * y_pred_c)
            total = K.sum(y_true_c) + K.sum(y_pred_c)
            union = total - intersection

            iou = (intersection + smooth) / (union + smooth)
            iou_loss_per_class.append(1 - iou)

        return K.mean(tf.stack(iou_loss_per_class))

    def combined_bce_iou_loss(y_true, y_pred):
        bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        iou = multiclass_iou_loss(y_true, y_pred)
        return bce_loss + iou

    model.build(keras.Input((batch_size, ) + INPUT_SHAPE))

#    opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    opt = keras.optimizers.Adam(learning_rate=0.01)
    model.compile(
        #loss=DiceLoss(),
        #loss='categorical_crossentropy',
        loss=combined_bce_iou_loss,
        optimizer=opt,
        metrics=[
            keras.metrics.OneHotMeanIoU(
                num_classes=CLASSES, sparse_y_pred=False
            ),
            keras.metrics.CategoricalAccuracy(),
        ],
    )

    print("Creating datagen")

    print("Input shape", model.inputs[0].shape[1:3])
    print("Inputs", model.inputs)

    if train_data is None:
        train_dir = os.path.join(data, 'train')
        train_data = gen_dataset(train_dir, MASK_SUBDIR, batch_size=batch_size, lab=lab, input_shape=input_shape[:2])
    if val_data is None:
        val_dir = os.path.join(data, 'val')
        val_data = gen_dataset(val_dir, MASK_SUBDIR, batch_size=batch_size, lab=lab, input_shape=input_shape[:2])
    # test_dir = os.path.join(data, 'test')
    #test_datagen = gen_dataset(TEST_DATA_PATH, MASK_SUBDIR, batch_size=batch_size, lab=lab)

    callbacks = [
        #keras.callbacks.EarlyStopping(patience=5),
        keras.callbacks.ModelCheckpoint(filepath='checkpoints/model_##name##.{epoch:02d}_##data##.keras'.replace("##name##", name).replace('##data##', os.path.basename(data))),
        keras.callbacks.TensorBoard(log_dir='./logs'),
        keras.callbacks.ModelCheckpoint(filepath='out/best_##name##_##data##.keras'.replace('##name##', name).replace('##data##', os.path.basename(data)), save_best_only=True, mode='max', monitor='val_one_hot_mean_io_u:')
    ]

    model.summary()

    print(f"Beginning training of model {name}")

    model.fit(train_data, epochs=epochs, callbacks=callbacks, validation_data=val_data)

    print("Training finished, starting test evaluation")

    result = model.evaluate(val_data)
    print(result)

def gen_dataset(path, mask_subdir, batch_size, lab, input_shape):
    x = keras.utils.image_dataset_from_directory(os.path.join(path, "images"),
                                                 batch_size=1,
                                                 image_size=input_shape[:2],
                                                 crop_to_aspect_ratio=True,
                                                 labels=None).map(lambda x0: x0 / 255)#.map(lambda x1: tf.expand_dims(x1, 0) if len(x1.shape) == 3 else x1)
    y = keras.utils.image_dataset_from_directory(os.path.join(path, mask_subdir),
                                                 batch_size=1,
                                                 image_size=input_shape[:2],
                                                 crop_to_aspect_ratio=True,
                                                 labels=None,
                                                 color_mode='grayscale').map(lambda y: to_categorical(y, num_classes=CLASSES))
    compare_datasets(os.path.join(path, "images"), os.path.join(path, mask_subdir))
    print("Dataset Sizes:", len(x), len(y))
    datagen = tf.data.Dataset.zip((x, y))
    if lab:
        datagen = datagen.map(
            lambda x, y: (transform_wrapper(x, target_size=INPUT_SHAPE[:2], rescale=True, smart_resize=True, lab=True), y)
        , num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    datagen = datagen.map(lambda x,y: (tf.squeeze(x, axis=0), tf.squeeze(y, axis=0))).batch(batch_size=batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    for s in datagen.take(1).as_numpy_iterator():
        print("X", s[0].shape, tf.reduce_max(s[0]).numpy())
        print("Y", s[1].shape, tf.reduce_max(s[1]).numpy())
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
        #(
        #    fcn_32_mobilenet(CLASSES, INPUT_SHAPE[0], INPUT_SHAPE[1]),
        #    "FCN32 Mobilenet",
        #    None,
        #    None
        #),
        #(
        #    vgg_segnet(CLASSES, INPUT_SHAPE[0], INPUT_SHAPE[1]),
        #    "VGG-Segnet",
        #    None,
        #    None
        #),
        (
           modellib.MaskRCNN(mode="training", model_dir=os.getcwd(), config=mrcnn_config_instance).keras_model,
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
        (
            pspnet_101(CLASSES, INPUT_SHAPE[0], INPUT_SHAPE[1]),
            "PSPNet",
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

def get_file_names_from_dataset(dataset):
    """
    Extracts the file names from a dataset loaded with image_dataset_from_directory.
    """
    file_names = set()
    for batch in dataset:
        file_paths = batch[0].numpy()
        for file_path in file_paths:
            file_name = os.path.basename(file_path.decode('utf-8'))
            file_names.add(file_name)
    return file_names

def compare_datasets(dataset1, dataset2):
    """
    Compares the file names of two datasets.
    """
    dataset1_files = set([os.path.basename(f) for f in os.listdir(dataset1) if f.endswith(('.png', '.jpg', '.jpeg'))])
    dataset2_files = set([os.path.basename(f) for f in os.listdir(dataset2) if f.endswith(('.png', '.jpg', '.jpeg'))])

    non_image_files1 = set([os.path.basename(f) for f in os.listdir(dataset1) if not f.endswith(('.png', '.jpg', '.jpeg'))])
    non_image_files2 = set([os.path.basename(f) for f in os.listdir(dataset2) if not f.endswith(('.png', '.jpg', '.jpeg'))])

    common_files = dataset1_files.intersection(dataset2_files)
    dataset1_only = dataset1_files - dataset2_files
    dataset2_only = dataset2_files - dataset1_files

    print(f"Number of files in dataset1: {len(dataset1_files)}")
    print(f"Number of files in dataset2: {len(dataset2_files)}")
    print(f"Number of common files: {len(common_files)}")
    print(f"Number of files only in dataset1: {len(dataset1_only)}")
    print(f"Number of files only in dataset2: {len(dataset2_only)}")
    print(f"Non-image files in dataset 1: #{len(non_image_files1)}: {non_image_files1}")
    print(f"Non-image files in dataset 2: #{len(non_image_files2)}: {non_image_files2}")

    if dataset1_only:
        print("\nFiles only in dataset1:")
        for file in dataset1_only:
            print(file)
    
    if dataset2_only:
        print("\nFiles only in dataset2:")
        for file in dataset2_only:
            print(file)

    if len(dataset1_only) == 0 and len(dataset2_only) == 0:
        print("\nThe datasets have the same file names.")
    else:
        print("\nThe datasets do not have the same file names.")

if __name__ == '__main__':
    main()