from keras_segmentation.train import train
import keras
import click
import os

@click.command()
@click.option('-m', '--model', type=str)
@click.option('-i', '--input', type=str)
@click.option('-a', '--augment', is_flag=True)
@click.option('-n', '--no-validate', is_flag=True)
@click.option('-c', '--classes', default=25)
def main(model, input, augment, no_validate, classes):
    train_images = os.path.join(input, "train", "images")
    train_anno = os.path.join(input, "train", "leaf_instances")

    val_images = os.path.join(input, "val", "images")
    val_anno = os.path.join(input, "val", "leaf_instances")

    callbacks = [
        #keras.callbacks.EarlyStopping(patience=5),
        keras.callbacks.ModelCheckpoint(filepath='checkpoints/model_##name##.{epoch:02d}_##data##.keras'.replace("##name##", model).replace('##data##', os.path.basename(input))),
        keras.callbacks.TensorBoard(log_dir='./logs'),
        keras.callbacks.ModelCheckpoint(filepath='out/best_##name##_##data##.keras'.replace('##name##', model).replace('##data##', os.path.basename(input)), save_best_only=True, mode='max', monitor='val_mean_io_u')
    ]

    train(model, 
          train_images, 
          train_anno,
          validate=not no_validate,
          val_images=val_images, 
          val_annotations=val_anno, 
          callbacks=callbacks, 
          do_augment=augment,
          n_classes=classes)

if __name__ == '__main__':
    main()