# Here's a codeblock just for fun. You should be able to upload an image here 
# and have it classified without crashing
import numpy as np
import camclient
import cv2
import tensorflow as tf
from skimage import color as skimage_color
import cai
import cai.layers
import os
from utils import transform_image
import keras

# CHKPT_PATH = "data/model/two-path-inception-v6-False-0.8-best_result.hdf5"
CHKPT_PATH = "data/model/0.8_best.hdf5"
MODEL_PATH = "data/model/two-path-inception-v2.8-False-0.2"
KERAS_PATH = os.path.join("data", "model", "two-path-inception-v2.8-False-0.2-best_result.keras")


TEST_DATA_PATH = "data/test_data"

def main():
    input_shape=(128, 128, 3)
    l_ratio = .6
    #model = cai.models.load_model(MODEL_PATH)
    # model = keras.models.load_model(KERAS_PATH)
    model = cai.models.load_kereas_model(KERAS_PATH)

    # model = tf.keras.models.load_model('data/model/0.8_best.hdf5',custom_objects={'CopyChannels': cai.layers.CopyChannels})
    model.summary()
    
    for i, class_name in enumerate(os.listdir(TEST_DATA_PATH)):
        for file in os.listdir(os.path.join(TEST_DATA_PATH, class_name)):
            img = cv2.imread(os.path.join(TEST_DATA_PATH, class_name, file))
            imm_array = transform_image(img, smart_resize=True, lab=True)

            cv2.imshow("Preview:", imm_array)
            cv2.waitKey(1)
            
            imm_array = np.expand_dims(imm_array, 0)

            predictions = model.predict(imm_array)
            prediction_score = tf.math.reduce_mean(tf.nn.softmax(predictions)).numpy()
            predicated_class = np.argmax(prediction_score)
            print(predicated_class, prediction_score, i, i == predicated_class)
            

if __name__ == '__main__':
    main()