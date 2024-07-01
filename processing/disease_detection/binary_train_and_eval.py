import cai.datasets
from schuler_two_branch.test_datagen import PlantLeafsDataGenBinary
import cai
import os
from tensorflow import keras
import keras.applications
import click
import tensorflow as tf

L_RATIO = .8
TWO_PATHS_SECOND_BLOCK = True
INPUT_SHAPE = (224, 224, 3)
CLASSES = 2

TRAIN_DATA_PATH = os.path.join("_data", "train_b")
VAL_DATA_PATH = os.path.join("_data", "val_b")
TEST_DATA_PATH = os.path.join("_data", "test_b")

def load_transform(paths):
    return cai.datasets.load_images_from_files(paths, target_size=INPUT_SHAPE[:2], lab=True, rescale=True, smart_resize=True)

def execute(model, name=None, lab=False, batch_size=32, workers=16):
    if name is None:
        name = type(model).__name__
    print(f"Starting training for {name}")

    opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    
    print("Creating datagen")
    # train_datagen = PlantLeafsDataGenBinary(TRAIN_DATA_PATH, transforms=[load_transform] if lab else None, batch_size=batch_size, workers=workers, use_multiprocessing=True)
    # val_datagen = PlantLeafsDataGenBinary(VAL_DATA_PATH, transforms=[load_transform] if lab else None, batch_size=batch_size, workers=workers, use_multiprocessing=True)
    # test_datagen = PlantLeafsDataGenBinary(TEST_DATA_PATH, transforms=[load_transform] if lab else None, batch_size=batch_size, workers=workers, use_multiprocessing=True)
    
    train_datagen = keras.utils.image_dataset_from_directory(TRAIN_DATA_PATH, batch_size=batch_size, image_size=INPUT_SHAPE[:2], crop_to_aspect_ratio=True, labels="inferred", label_mode="binary")
    train_datagen = train_datagen.map(lambda x, y: (tf.expand_dims(x, 0), y)).prefetch(tf.data.AUTOTUNE)
    val_datagen = keras.utils.image_dataset_from_directory(VAL_DATA_PATH, batch_size=batch_size, image_size=INPUT_SHAPE[:2], crop_to_aspect_ratio=True, labels="inferred", label_mode="binary")
    val_datagen = val_datagen.map(lambda x, y: (tf.expand_dims(x, 0), y)).prefetch(tf.data.AUTOTUNE)
    test_datagen = keras.utils.image_dataset_from_directory(TEST_DATA_PATH, batch_size=batch_size, image_size=INPUT_SHAPE[:2], crop_to_aspect_ratio=True, labels="inferred", label_mode="binary")
    test_datagen = test_datagen.map(lambda x, y: (tf.expand_dims(x, 0), y)).prefetch(tf.data.AUTOTUNE)

    test = train_datagen.take(5).as_numpy_iterator()
    for el in test:
        print(el.shape)

    print("Dataset sizes [train, val, test]", len(train_datagen), len(val_datagen), len(test_datagen))

    callbacks = [
        keras.callbacks.EarlyStopping(patience=2),
        keras.callbacks.ModelCheckpoint(filepath='checkpoints/model##name##.{epoch:02d}.keras'.replace("##name##", name)),
        keras.callbacks.TensorBoard(log_dir='./logs'),
        keras.callbacks.ModelCheckpoint(filepath='out/best##name##.keras'.replace('##name##', name), save_best_only=True, mode='max', monitor='val_accuracy')
    ]
    print(f"Beginning training of model {name}")

    model.fit(train_datagen, epochs=15, callbacks=callbacks, validation_data=val_datagen)

    print("Training finished, starting test evaluation")

    result = model.eval(test_datagen)
    print(result)

@click.command()
@click.option("-w", "--workers", type=int)
@click.option("-b", "--batch_size", type=int)
def main(workers, batch_size):
    models = [
        (
            keras.applications.ResNet152V2(
                include_top = True,
                input_shape=INPUT_SHAPE,
                classes=CLASSES,
                weights=None,
            ),
            "ResNet152V2"
        ),
        (
            keras.applications.InceptionV3(
                include_top=True,
                input_shape=INPUT_SHAPE,
                classes=CLASSES,
                weights=None,
            ),
            "InceptionV3"
        ),
        (
            keras.applications.MobileNetV3Large(
                include_top=True,
                input_shape=INPUT_SHAPE,
                classes=CLASSES,
                weights=None,
            ),
            "MobileNetV3Large"
        ),
        (
            keras.applications.ConvNeXtLarge(
                include_top=True,
                input_shape=INPUT_SHAPE,
                classes=CLASSES,
                weights=None,
            ),
            "ConvNeXtLarge"
        )
    ]

    for model, name in models:
        for lab in [True, False]:
            execute(model, f"{name}_{'lab' if lab else 'rgb'}", lab, workers=workers, batch_size=batch_size)

if __name__ == '__main__':
    main()