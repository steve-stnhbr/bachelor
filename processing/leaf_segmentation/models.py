from keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D, UpSampling2D, Flatten, Dense
from keras.models import Sequential

def plrs_net(img_shape=(224, 224, 3), n_classes=1000):
	'''
	Created from paper described in https://www.emerald.com/insight/content/doi/10.1108/IJIUS-08-2021-0100/full/html
	'''
	# Define the model
	model = Sequential()
	# Encoder part
	model.add(Conv2D(64, (3, 3), padding='same'), input_shape=img_shape)
	model.add(BatchNormalization())
	model.add(ReLU())
	model.add(MaxPooling2D(pool_size=(2, 2)))

	# Decoder part
	model.add(UpSampling2D(size=(2, 2)))
	model.add(Conv2D(3, (3, 3), padding='same'))  # Assuming we want to go back to 3 channels
	model.add(BatchNormalization())
	model.add(ReLU())

	# Classification part
	model.add(Flatten())
	model.add(Dense(n_classes, activation='softmax'))