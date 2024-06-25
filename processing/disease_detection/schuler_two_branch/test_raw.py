
from keras.preprocessing.image import load_img, img_to_array
import cai
import os
import numpy as np
import gc
import glob
import random
from sklearn.utils import class_weight
from keras.utils import to_categorical
import tensorflow as tf

TEST_DATA_PATH = "data/test_data"
MODEL_PATH = "checkpoints/model.11.keras"

def read_from_paths(paths):
    x=[]
    for path in paths:
      img = load_img(path, target_size=(224,224))
      img = img_to_array(img)
      x.append(img)
    return x

def load_data(path=TEST_DATA_PATH, lab=True, verbose=True, bipolar=False):
    classes = os.listdir(path)
    classes = sorted(classes)
    train_path = []
    train_x,train_y = [],[]
    for i,_class in enumerate(classes):
      paths = glob.glob(os.path.join(path,_class,"*"))
      paths = [n for n in paths if n.endswith(".JPG") or n.endswith(".jpg")]
      random.shuffle(paths)
      cat_total = len(paths)
      train_path.extend(paths[:int(cat_total*0.6)])
      train_y.extend([i]*int(cat_total*0.6))
    train_x = np.array(read_from_paths(TEST_DATA_PATH), dtype='float16')
    if (lab):
        # LAB datasets are cached
        if (verbose):
            print("Converting RGB to LAB")
        train_x /= 255
        if (verbose):
            print("Converting training.")
        cai.datasets.skimage_rgb2lab_a(train_x,  verbose)
        if (bipolar):
            # JP prefers bipolar input [-2,+2]
            train_x[:,:,:,0:3] /= [25, 50, 50]
            train_x[:,:,:,0] -= 2
        else:
            train_x[:,:,:,0:3] /= [100, 200, 200]
            train_x[:,:,:,1:3] += 0.
    else:
        if (verbose):
            print("Loading RGB.")
        if (bipolar):
            train_x /= 6
            train_x -= 2
        else:
            train_x /= 255

    if (verbose):
            for channel in range(0, train_x.shape[3]):
                sub_matrix = train_x[:,:,:,channel]
                print('Channel ', channel, ' min:', np.min(sub_matrix), ' max:', np.max(sub_matrix))
    #calculate class weight
    classweight = class_weight.compute_class_weight('balanced', np.unique(train_y), train_y)

    #convert to categorical
    train_y = to_categorical(train_y, 38)
    return train_x,train_y,classweight,classes

def load_file(path, lab=True, verbose=True, bipolar=False):
    train_x = np.array(read_from_paths([path]), dtype='float16')

    if (lab):
        # LAB datasets are cached
        if (verbose):
            print("Converting RGB to LAB")         
        train_x /= 255
        if (verbose):
            print("Converting training.")
        cai.datasets.skimage_rgb2lab_a(train_x,  verbose)
        if (bipolar):
            # JP prefers bipolar input [-2,+2]
            train_x[:,:,:,0:3] /= [25, 50, 50]
            train_x[:,:,:,0] -= 2
        else:
            train_x[:,:,:,0:3] /= [100, 200, 200]
            train_x[:,:,:,1:3] += 0.
    else:
        if (verbose):
            print("Loading RGB.")
        if (bipolar):
            train_x /= 6
            train_x -= 2
        else:
            train_x /= 255
    
    return train_x

def main():
    # load model
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'CopyChannels': cai.layers.CopyChannels})

    model.summary()
    total = 0
    rightly_predicted = 0
    # iterate over all data classes
    for i, class_name in enumerate(os.listdir(TEST_DATA_PATH)):
        # iterate over all files in test class
        for file in os.listdir(os.path.join(TEST_DATA_PATH, class_name)):
            # load and convert file
            imm_array = load_file(os.path.join(TEST_DATA_PATH, class_name, file))
            # create prediction
            predictions = model.predict(imm_array)
            
            predicted = np.argmax(predictions, 0)
            
            if predicted == i:
                rightly_predicted += 1

            total += 1
    
    print("Total: {}, Correct: {}, Accuracy: {}".format(total, rightly_predicted, rightly_predicted/total))

if __name__ == '__main__':
    main()