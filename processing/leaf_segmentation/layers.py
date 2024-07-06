import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Lambda

# Function for max pooling with argmax
def max_pool_with_argmax(net, ksize, strides):
    assert isinstance(ksize, list) or isinstance(ksize, int)
    assert isinstance(strides, list) or isinstance(strides, int)

    ksize = ksize if isinstance(ksize, list) else [1, ksize, ksize, 1]
    strides = strides if isinstance(strides, list) else [1, strides, strides, 1]

    with tf.name_scope('MaxPoolArgMax'):
        net, mask = tf.nn.max_pool_with_argmax(
            net,
            ksize=ksize,
            strides=strides,
            padding='SAME')
        return net, mask

class MaxPoolingWithArgmax2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def call(self, inputs, **kwargs):
        pool_size = self.pool_size
        strides = self.strides
        ksize = [1, pool_size[0], pool_size[1], 1]
        strides = [1, strides[0], strides[1], 1]
        output, argmax = Lambda(lambda x: max_pool_with_argmax(x, ksize, strides))(inputs)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [
            dim // ratio[idx]
            if dim is not None else None
            for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]

# Example usage:
if __name__ == '__main__':
    import numpy as np

    input = keras.layers.Input((4, 4, 3))
    (e, m) = MaxPoolingWithArgmax2D()(input)
    model = keras.Model(inputs=input, outputs=[e, m])
    model.compile(optimizer="adam", loss='categorical_crossentropy')
    model.summary()
    x = np.random.randint(0, 100, (3, 4, 4, 3))
    m = model.predict(x)
    print(x[0])
    print(m[0])
