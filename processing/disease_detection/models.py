# sourced from https://github.com/eweill/keras-deepcv/blob/master/models/classification/alexnet.py
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2

def alexnet_model(img_shape=(224, 224, 3), n_classes=10, l2_reg=0.,
	weights=None):

	# Initialize model
	alexnet = Sequential()

	# Layer 1
	alexnet.add(Conv2D(96, (11, 11), input_shape=img_shape,
		padding='same', kernel_regularizer=l2(l2_reg)))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 2
	alexnet.add(Conv2D(256, (5, 5), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 3
	alexnet.add(ZeroPadding2D((1, 1)))
	alexnet.add(Conv2D(512, (3, 3), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 4
	alexnet.add(ZeroPadding2D((1, 1)))
	alexnet.add(Conv2D(1024, (3, 3), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))

	# Layer 5
	alexnet.add(ZeroPadding2D((1, 1)))
	alexnet.add(Conv2D(1024, (3, 3), padding='same'))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 6
	alexnet.add(Flatten())
	alexnet.add(Dense(3072))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(Dropout(0.5))

	# Layer 7
	alexnet.add(Dense(4096))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('relu'))
	alexnet.add(Dropout(0.5))

	# Layer 8
	alexnet.add(Dense(n_classes))
	alexnet.add(BatchNormalization())
	alexnet.add(Activation('softmax'))

	if weights is not None:
		alexnet.load_weights(weights)

	return alexnet


def lenet_model(img_shape=(28, 28, 1), n_classes=10, l2_reg=0.,
	weights=None):

	# Initialize model
	lenet = Sequential()

	# 2 sets of CRP (Convolution, RELU, Pooling)
	lenet.add(Conv2D(20, (5, 5), padding="same",
		input_shape=img_shape, kernel_regularizer=l2(l2_reg)))
	lenet.add(Activation("relu"))
	lenet.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	lenet.add(Conv2D(50, (5, 5), padding="same",
		kernel_regularizer=l2(l2_reg)))
	lenet.add(Activation("relu"))
	lenet.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	# Fully connected layers (w/ RELU)
	lenet.add(Flatten())
	lenet.add(Dense(500, kernel_regularizer=l2(l2_reg)))
	lenet.add(Activation("relu"))

	# Softmax (for classification)
	lenet.add(Dense(n_classes, kernel_regularizer=l2(l2_reg)))
	lenet.add(Activation("softmax"))

	if weights is not None:
		lenet.load_weights(weights)

	# Return the constructed network
	return lenet

def vgg16_model(img_shape=(224, 224, 3), n_classes=1000, l2_reg=0.,
	weights=None):

	# Initialize model
	vgg16 = Sequential()

	# Layer 1 & 2
	vgg16.add(Conv2D(64, (3, 3), padding='same',
		input_shape=img_shape, kernel_regularizer=l2(l2_reg)))
	vgg16.add(Activation('relu'))
	vgg16.add(ZeroPadding2D((1, 1)))
	vgg16.add(Conv2D(64, (3, 3), padding='same'))
	vgg16.add(Activation('relu'))
	vgg16.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 3 & 4
	vgg16.add(ZeroPadding2D((1, 1)))
	vgg16.add(Conv2D(128, (3, 3), padding='same'))
	vgg16.add(Activation('relu'))
	vgg16.add(ZeroPadding2D((1, 1)))
	vgg16.add(Conv2D(128, (3, 3), padding='same'))
	vgg16.add(Activation('relu'))
	vgg16.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 5, 6, & 7
	vgg16.add(ZeroPadding2D((1, 1)))
	vgg16.add(Conv2D(256, (3, 3), padding='same'))
	vgg16.add(Activation('relu'))
	vgg16.add(ZeroPadding2D((1, 1)))
	vgg16.add(Conv2D(256, (3, 3), padding='same'))
	vgg16.add(Activation('relu'))
	vgg16.add(ZeroPadding2D((1, 1)))
	vgg16.add(Conv2D(256, (3, 3), padding='same'))
	vgg16.add(Activation('relu'))
	vgg16.add(MaxPooling2D(pool_size=(2, 2)))

	# Layers 8, 9, & 10
	vgg16.add(ZeroPadding2D((1, 1)))
	vgg16.add(Conv2D(512, (3, 3), padding='same'))
	vgg16.add(Activation('relu'))
	vgg16.add(ZeroPadding2D((1, 1)))
	vgg16.add(Conv2D(512, (3, 3), padding='same'))
	vgg16.add(Activation('relu'))
	vgg16.add(ZeroPadding2D((1, 1)))
	vgg16.add(Conv2D(512, (3, 3), padding='same'))
	vgg16.add(Activation('relu'))
	vgg16.add(MaxPooling2D(pool_size=(2, 2)))

	# Layers 11, 12, & 13
	vgg16.add(ZeroPadding2D((1, 1)))
	vgg16.add(Conv2D(512, (3, 3), padding='same'))
	vgg16.add(Activation('relu'))
	vgg16.add(ZeroPadding2D((1, 1)))
	vgg16.add(Conv2D(512, (3, 3), padding='same'))
	vgg16.add(Activation('relu'))
	vgg16.add(ZeroPadding2D((1, 1)))
	vgg16.add(Conv2D(512, (3, 3), padding='same'))
	vgg16.add(Activation('relu'))
	vgg16.add(MaxPooling2D(pool_size=(2, 2)))

	# Layers 14, 15, & 16
	vgg16.add(Flatten())
	vgg16.add(Dense(4096))
	vgg16.add(Activation('relu'))
	vgg16.add(Dropout(0.5))
	vgg16.add(Dense(4096))
	vgg16.add(Activation('relu'))
	vgg16.add(Dropout(0.5))
	vgg16.add(Dense(n_classes))
	vgg16.add(Activation('softmax'))

	if weights is not None:
		vgg16.load_weights(weights)

	return vgg16

def vgg19_model(img_shape=(224, 224, 3), n_classes=1000, l2_reg=0.,
	weights=None):

	# Initialize model
	vgg19 = Sequential()

	# Layer 1 & 2
	vgg19.add(Conv2D(64, (3, 3), padding='same',
		input_shape=img_shape, kernel_regularizer=l2(l2_reg)))
	vgg19.add(Activation('relu'))
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(64, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 3 & 4
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(128, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(128, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(MaxPooling2D(pool_size=(2, 2)))

	# Layer 5, 6, 7, & 8
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(256, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(256, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(256, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(256, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(MaxPooling2D(pool_size=(2, 2)))

	# Layers 9, 10, 11, & 12
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(512, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(512, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(512, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(512, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(MaxPooling2D(pool_size=(2, 2)))

	# Layers 13, 14, 15, & 16
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(512, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(512, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(512, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(ZeroPadding2D((1, 1)))
	vgg19.add(Conv2D(512, (3, 3), padding='same'))
	vgg19.add(Activation('relu'))
	vgg19.add(MaxPooling2D(pool_size=(2, 2)))

	# Layers 17, 18, & 19
	vgg19.add(Flatten())
	vgg19.add(Dense(4096))
	vgg19.add(Activation('relu'))
	vgg19.add(Dropout(0.5))
	vgg19.add(Dense(4096))
	vgg19.add(Activation('relu'))
	vgg19.add(Dropout(0.5))
	vgg19.add(Dense(n_classes))
	vgg19.add(Activation('softmax'))

	if weights is not None:
		vgg19.load_weights(weights)

	return vgg19