import numpy as np
import keras
from keras.models import *
from keras.layers import *
import keras.backend as K
import tensorflow as tf

from .config import IMAGE_ORDERING
from .model_utils import get_segmentation_model, resize_image
from .vgg16 import get_vgg_encoder
from .basic_models import vanilla_encoder
from .resnet50 import get_resnet50_encoder


if IMAGE_ORDERING == 'channels_first':
    MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
    MERGE_AXIS = -1


def pool_block(feats, pool_factor):

    if IMAGE_ORDERING == 'channels_first':
        h = feats.shape[2]
        w = feats.shape[3]
    elif IMAGE_ORDERING == 'channels_last':
        h = feats.shape[1]
        w = feats.shape[2]

    pool_size = strides = [
        int(np.round(float(h) / pool_factor)),
        int(np.round(float(w) / pool_factor))]

    x = AveragePooling2D(pool_size, data_format=IMAGE_ORDERING,
                         strides=strides, padding='same')(feats)
    x = Conv2D(512, (1, 1), data_format=IMAGE_ORDERING,
               padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    #x = resize_image(x, strides, data_format=IMAGE_ORDERING)
    x = ResizeImagesByFactor(strides[0], strides[1], data_format=IMAGE_ORDERING)(x)

    return x


def _pspnet(n_classes, encoder,  input_height=384, input_width=576, channels=3):

    assert input_height % 192 == 0
    assert input_width % 192 == 0

    img_input, levels = encoder(
        input_height=input_height,  input_width=input_width, channels=channels)
    [f1, f2, f3, f4, f5] = levels

    o = f5

    pool_factors = [1, 2, 3, 6]
    pool_outs = [o]

    for p in pool_factors:
        pooled = pool_block(o, p)
        pool_outs.append(pooled)

    print([pool.shape for pool in pool_outs])

    o = Concatenate(axis=MERGE_AXIS)(pool_outs)

    o = Conv2D(512, (1, 1), data_format=IMAGE_ORDERING, use_bias=False , name="seg_feats" )(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)

    o = Conv2D(n_classes, (3, 3), data_format=IMAGE_ORDERING,
               padding='same')(o)
    # o = resize_image(o, (8, 8), data_format=IMAGE_ORDERING)
    o = ResizeImagesByFactor(8, 8, data_format=IMAGE_ORDERING)(o)
    o = ResizeImagesByFactor(4, 4, data_format=IMAGE_ORDERING)(o)

    model = get_segmentation_model(img_input, o)
    return model


def pspnet(n_classes,  input_height=384, input_width=576, channels=3):

    model = _pspnet(n_classes, vanilla_encoder,
                    input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "pspnet"
    return model


def vgg_pspnet(n_classes,  input_height=384, input_width=576, channels=3):

    model = _pspnet(n_classes, get_vgg_encoder,
                    input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "vgg_pspnet"
    return model


def resnet50_pspnet(n_classes,  input_height=384, input_width=576, channels=3):

    model = _pspnet(n_classes, get_resnet50_encoder,
                    input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "resnet50_pspnet"
    return model


def pspnet_50(n_classes,  input_height=473, input_width=473, channels=3):
    from ._pspnet_2 import _build_pspnet

    nb_classes = n_classes
    resnet_layers = 50
    input_shape = (input_height, input_width)
    model = _build_pspnet(nb_classes=nb_classes,
                          resnet_layers=resnet_layers,
                          input_shape=input_shape, channels=channels)
    model.model_name = "pspnet_50"
    return model


def pspnet_101(n_classes,  input_height=473, input_width=473, channels=3):
    from ._pspnet_2 import _build_pspnet

    nb_classes = n_classes
    resnet_layers = 101
    input_shape = (input_height, input_width)
    model = _build_pspnet(nb_classes=nb_classes,
                          resnet_layers=resnet_layers,
                          input_shape=input_shape, channels=channels)
    model.model_name = "pspnet_101"
    return model


# def mobilenet_pspnet( n_classes ,  input_height=224, input_width=224 ):

# 	model =  _pspnet(n_classes, get_mobilenet_encoder,
#                    input_height=input_height, input_width=input_width)
# 	model.model_name = "mobilenet_pspnet"
# 	return model


class ResizeImagesByFactor(Layer):
    def __init__(self, height_factor, width_factor, data_format="channels_last", interpolation="bilinear", **kwargs):
        super(ResizeImagesByFactor, self).__init__(**kwargs)
        self.height_factor = height_factor
        self.width_factor = width_factor
        self.data_format = data_format
        self.interpolation = interpolation

    def build(self, input_shape):
        super(ResizeImagesByFactor, self).build(input_shape)

    def call(self, inputs):
        if self.data_format == "channels_first":
            input_shape = inputs.shape
            original_height = tf.cast(input_shape[2], tf.float32)
            original_width = tf.cast(input_shape[3], tf.float32)
        elif self.data_format == "channels_last":
            input_shape = inputs.shape
            original_height = tf.cast(input_shape[1], tf.float32)
            original_width = tf.cast(input_shape[2], tf.float32)
        else:
            raise ValueError(f"Invalid `data_format` argument: {self.data_format}")

        new_height = tf.cast(original_height * self.height_factor, tf.int32)
        new_width = tf.cast(original_width * self.width_factor, tf.int32)

        resized = tf.image.resize(inputs, [new_height, new_width], method=self.interpolation)

        if self.data_format == "channels_first":
            resized = Permute((0, 2, 3, 1))(resized)

        return resized

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            return (input_shape[0], input_shape[1], None, None)
        elif self.data_format == "channels_last":
            return (input_shape[0], None, None, input_shape[3])
        else:
            raise ValueError(f"Invalid `data_format` argument: {self.data_format}")


if __name__ == '__main__':

    m = _pspnet(101, vanilla_encoder)
    # m = _pspnet( 101 , get_mobilenet_encoder ,True , 224 , 224  )
    m = _pspnet(101, get_vgg_encoder)
    m = _pspnet(101, get_resnet50_encoder)
