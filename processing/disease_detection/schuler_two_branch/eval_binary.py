import pandas as pd
import os
from tqdm import tqdm
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
from utils import transform_image
import cv2
import tensorflow as tf
from itertools import cycle

VERBOSE = False
MODEL_PATH = "data/model/model.12.keras"
DATA_PATH = "../_data/test"
INPUT_SHAPE = (224, 224, 3)

def read_from_paths(paths):
    x=[]
    for path in paths:
      img = load_img(path, target_size=(224,224))
      img = img_to_array(img)
      x.append(img)
    return x

def load_data(path=DATA_PATH, lab=True, verbose=True, bipolar=False):
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
    train_x = np.array(read_from_paths(DATA_PATH), dtype='float16')
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

def get_image(file):
    return cai.datasets.load_images_from_files([file], target_size=INPUT_SHAPE[:2], lab=True, rescale=True, smart_resize=True)

    image = cv2.imread(file)
    image = transform_image(image, smart_resize=True, lab=True, rescale=True)
    image = np.expand_dims(image, 0)
    return image


def main():
    # load model
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'CopyChannels': cai.layers.CopyChannels})

    model.summary()

    num_correct_pred = 0
    num_wrong_pred = 0
    eval = [(clazz, "healthy" in clazz.lower(), os.path.join(DATA_PATH, clazz, file)) for clazz in os.listdir(DATA_PATH) for file in os.listdir(os.path.join(DATA_PATH, clazz))]

    diseases = list(set(eval[0]))
    data = map(lambda x: [x, 0, 0, 0], diseases)
    df = pd.DataFrame(data, columns=['disease', 'amount', 'predicted', 'correct'])

    try:
        for label, image_file in tqdm(eval):
            image = get_image(image_file)
            if VERBOSE:
                print("Image size: {}".format(image.size()))
            predictions = model.predict(image)
            predicted_label = np.argmax(predictions)
            if VERBOSE:
                print("Prediction: {}".format(predicted_label))
                print("Actual: {}".format(label))
            df.at[label, 'amount'] += 1
            df.at[predicted_label, 'predicted'] += 1
            if predicted_label != label:
                num_wrong_pred += 1
            else:
                num_correct_pred += 1
                df.at[label, 'correct'] += 1
    except KeyboardInterrupt:
        print("Accuracy: ", num_correct_pred / (num_correct_pred + num_wrong_pred))
        df.to_csv("./out/result.csv")


    print("Accuracy: ", num_correct_pred / (num_correct_pred + num_wrong_pred))
    df.to_csv("./out/result.csv")
    print(df)

if __name__ == '__main__':
    main()