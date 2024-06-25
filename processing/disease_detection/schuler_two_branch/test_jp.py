# Here's a codeblock just for fun. You should be able to upload an image here 
# and have it classified without crashing
import numpy as np
import cv2
import tensorflow as tf
from skimage import color as skimage_color
import cai
import cai.layers
import os
from utils import transform_image
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# CHKPT_PATH = "data/model/two-path-inception-v6-False-0.8-best_result.hdf5"
CHKPT_PATH = "data/model/0.8_best.hdf5"
MODEL_PATH = "data/model/two-path-inception-v2.8-False-0.2"
KERAS_PATH = os.path.join("checkpoints", "model.11.keras")

TEST_DATA_PATH = "data/test"

def main():
    input_shape=(128, 128, 3)
    model = cai.models.load_model(MODEL_PATH)

    # model = tf.keras.models.load_model('data/model/0.8_best.hdf5',custom_objects={'CopyChannels': cai.layers.CopyChannels})
    model.summary()
    
    for i, class_name in enumerate(os.listdir(TEST_DATA_PATH)):
        for file in os.listdir(os.path.join(TEST_DATA_PATH, class_name)):
            img = load_img(os.path.join(TEST_DATA_PATH, class_name, file), target_size=input_shape[:2])
            imm_array = transform_image(img, smart_resize=True, lab=True)

            predictions = model.predict(imm_array)
            prediction_score = tf.math.reduce_mean(tf.nn.softmax(predictions)).numpy()
            predicated_class = np.argmax(prediction_score)
            print(predictions, np.argmax(predictions))
            print(predicated_class, prediction_score, i, i == predicated_class)
            
def transform_image(img, lab, bipolar=False, verbose=True, smart_resize=False):
    img = np.array(img, dtype='float16')
    img = np.expand_dims(img, 0)
    if (lab):
        if (verbose):
            print("Converting RGB to LAB")
        img /= 255
        if (verbose):
            print("Converting training.")
        cai.datasets.skimage_rgb2lab_a(img,  verbose)
        if (bipolar):
            # JP prefers bipolar input [-2,+2]
            img[:,:,:,0:3] /= [25, 50, 50]
        else:
            img[:,:,:,0:3] /= [100, 200, 200]
    else:
        if (verbose):
            print("Loading RGB.")
        if (bipolar):
            img /= 64
            img -= 2
        else:
            img /= 255
    return img

if __name__ == '__main__':
    main()