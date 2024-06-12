import cai.datasets
from test_datagen import PlantLeafsDataGen
import cai
import os
import tensorflow as tf
from tensorflow import keras

L_RATIO = .8
TWO_PATHS_SECOND_BLOCK = True
INPUT_SHAPE = (224, 224, 3)

DATA_PATH = os.path.join("2.8", "plant_leaf", "Plant_leave_diseases_dataset_without_augmentation")

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
    
    datagen = PlantLeafsDataGen(DATA_PATH, transforms=[load_transform])

    model.fit(datagen, epochs=20)
    
if __name__ == '__main__':
    main()