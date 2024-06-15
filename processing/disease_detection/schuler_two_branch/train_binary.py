import cai.datasets
from test_datagen import PlantLeafsDataGenBinary
import cai
import os
import tensorflow as tf
from tensorflow import keras
from test_raw import load_file
import numpy as np

L_RATIO = .8
TWO_PATHS_SECOND_BLOCK = True
INPUT_SHAPE = (224, 224, 3)

TRAIN_DATA_PATH = os.path.join("2.8", "plant_leaf", "Plant_leave_diseases_dataset_without_augmentation")
VAL_DATA_PATH = os.path.join("data", "test_data")

def load_transform(paths):
    return cai.datasets.load_images_from_files(paths, target_size=INPUT_SHAPE[:2], lab=True, rescale=True, smart_resize=True)

def main():
    print(tf.config.list_physical_devices())
    model = cai.models.compiled_inception_v3(classes=2)
    
    datagen = PlantLeafsDataGenBinary(TRAIN_DATA_PATH, transforms=[load_transform], batch_size=32, workers=9, use_multiprocessing=True)

    callbacks = [
        keras.callbacks.EarlyStopping(patience=2),
        keras.callbacks.ModelCheckpoint(filepath='checkpoints/model.{epoch:02d}.keras'),
        keras.callbacks.TensorBoard(log_dir='./logs'),
    ]

    model.fit(datagen, epochs=4, callbacks=callbacks)

    print("Training finished, starting evaluation")


    for i, class_name in enumerate(os.listdir(VAL_DATA_PATH)):
        # iterate over all files in test class
        for file in os.listdir(os.path.join(VAL_DATA_PATH, class_name)):
            # load and convert file
            imm_array = load_file(os.path.join(VAL_DATA_PATH, class_name, file))
            # create prediction
            predictions = model.predict(imm_array)
            # calculate prediction score
            prediction_score = tf.math.reduce_mean(tf.nn.softmax(predictions)).numpy()
            # determine class with highest confidence
            predicated_class = np.argmax(prediction_score)
            clazz = 1 if "healthy" in class_name else 0
            print(predictions, np.argmax(predictions, axis=1), predicated_class, prediction_score, clazz, clazz == predicated_class)
    
    
if __name__ == '__main__':
    main()