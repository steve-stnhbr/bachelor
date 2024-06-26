import cai.datasets
from test_datagen import PlantLeafsDataGen
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

    model = keras.models.load_model("data/model/0.8_best.hdf5", custom_objects={'CopyChannels': cai.layers.CopyChannels})
    opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'])
    
    datagen = PlantLeafsDataGen(TRAIN_DATA_PATH, transforms=[load_transform], batch_size=128, workers=9, use_multiprocessing=True)

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
            imm_array = load_transform([os.path.join(VAL_DATA_PATH, class_name, file)])
            # create prediction
            predictions = model.predict(imm_array)
            print(predictions, np.argmax(predictions, axis=1))
            # calculate prediction score
            prediction_score = tf.math.reduce_mean(tf.nn.softmax(predictions)).numpy()
            # determine class with highest confidence
            predicated_class = np.argmax(prediction_score)
            print(predicated_class, prediction_score, i, i == predicated_class)
    
    
if __name__ == '__main__':
    main()