import json
import os

from keras_segmentation.data_utils.data_loader import image_segmentation_generator, \
    verify_segmentation_dataset
import keras_segmentation.models.all_models as all_models
import six
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import glob
import sys
import keras
import click

def find_latest_checkpoint(checkpoints_path, fail_safe=True):

    # This is legacy code, there should always be a "checkpoint" file in your directory

    def get_epoch_number_from_path(path):
        return path.replace(checkpoints_path, "").strip(".")

    # Get all matching files
    all_checkpoint_files = glob.glob(checkpoints_path + ".*")
    if len(all_checkpoint_files) == 0:
        all_checkpoint_files = glob.glob(checkpoints_path + "*.*")
    all_checkpoint_files = [ff.replace(".index", "") for ff in
                            all_checkpoint_files]  # to make it work for newer versions of keras
    # Filter out entries where the epoc_number part is pure number
    all_checkpoint_files = list(filter(lambda f: get_epoch_number_from_path(f)
                                       .isdigit(), all_checkpoint_files))
    if not len(all_checkpoint_files):
        # The glob list is empty, don't have a checkpoints_path
        if not fail_safe:
            raise ValueError("Checkpoint path {0} invalid"
                             .format(checkpoints_path))
        else:
            return None

    # Find the checkpoint file with the maximum epoch
    latest_epoch_checkpoint = max(all_checkpoint_files,
                                  key=lambda f:
                                  int(get_epoch_number_from_path(f)))

    return latest_epoch_checkpoint

def masked_categorical_crossentropy(gt, pr):
    from keras.losses import categorical_crossentropy
    
    # Create a mask where the gt labels are not equal to the background (assuming 0 is the background class)
    mask = tf.reduce_max(gt, axis=-1)  # Reduce max along the last axis to get [batch_size, height, width]
    mask = tf.cast(tf.not_equal(mask, 0), tf.float32)  # Convert mask to float and boolean not equal to 0
    
    # Compute the categorical crossentropy
    loss = categorical_crossentropy(gt, pr)
    
    # Apply mask to the loss
    loss = loss * mask
    
    return tf.reduce_mean(loss)

class CheckpointsCallback(Callback):
    def __init__(self, checkpoints_path):
        self.checkpoints_path = checkpoints_path

    def on_epoch_end(self, epoch, logs=None):
        if self.checkpoints_path is not None:
            self.model.save_weights(self.checkpoints_path + "." + str(epoch))
            print("saved ", self.checkpoints_path + "." + str(epoch))


def train(model,
          train_images,
          train_annotations,
          input_height=None,
          input_width=None,
          n_classes=None,
          verify_dataset=True,
          checkpoints_path=None,
          epochs=5,
          batch_size=2,
          validate=False,
          val_images=None,
          val_annotations=None,
          val_batch_size=2,
          auto_resume_checkpoint=False,
          load_weights=None,
          steps_per_epoch=512,
          val_steps_per_epoch=512,
          gen_use_multiprocessing=False,
          ignore_zero_class=False,
          optimizer_name='adam',
          do_augment=False,
          augmentation_name="aug_all",
          callbacks=None,
          custom_augmentation=None,
          other_inputs_paths=None,
          preprocessing=None,
          read_image_type=1  # cv2.IMREAD_COLOR = 1 (rgb),
                             # cv2.IMREAD_GRAYSCALE = 0,
                             # cv2.IMREAD_UNCHANGED = -1 (4 channels like RGBA)
         ):
    from keras_segmentation.models.all_models import model_from_name
    # check if user gives model name instead of the model object
    if isinstance(model, six.string_types):
        # create the model from the name
        assert (n_classes is not None), "Please provide the n_classes"
        if (input_height is not None) and (input_width is not None):
            model = model_from_name[model](
                n_classes, input_height=input_height, input_width=input_width)
        else:
            model = model_from_name[model](n_classes)

    model.summary()

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    output_height = input_height
    output_width = input_width

    if validate:
        assert val_images is not None
        assert val_annotations is not None

    if optimizer_name is not None:

        if ignore_zero_class:
            loss_k = masked_categorical_crossentropy
        else:
            loss_k = 'categorical_crossentropy'

        model.compile(loss=loss_k,
                      optimizer=optimizer_name,
                      metrics=['accuracy'])

    if checkpoints_path is not None:
        config_file = checkpoints_path + "_config.json"
        dir_name = os.path.dirname(config_file)

        if ( not os.path.exists(dir_name) )  and len( dir_name ) > 0 :
            os.makedirs(dir_name)

        with open(config_file, "w") as f:
            json.dump({
                "model_class": model.model_name,
                "n_classes": n_classes,
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width
            }, f)

    if load_weights is not None and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    initial_epoch = 0

    if auto_resume_checkpoint and (checkpoints_path is not None):
        latest_checkpoint = find_latest_checkpoint(checkpoints_path)
        if latest_checkpoint is not None:
            print("Loading the weights from latest checkpoint ",
                  latest_checkpoint)
            model.load_weights(latest_checkpoint)

            initial_epoch = int(latest_checkpoint.split('.')[-1])

    if verify_dataset:
        print("Verifying training dataset")
        verified = verify_segmentation_dataset(train_images,
                                               train_annotations,
                                               n_classes)
        assert verified
        if validate:
            print("Verifying validation dataset")
            verified = verify_segmentation_dataset(val_images,
                                                   val_annotations,
                                                   n_classes)
            assert verified

    print(input_width, input_height, output_width, output_height)

    train_gen = image_segmentation_generator(
        train_images, train_annotations,  batch_size,  n_classes,
        input_height, input_width, output_height, output_width,
        do_augment=do_augment, augmentation_name=augmentation_name,
        custom_augmentation=custom_augmentation, other_inputs_paths=other_inputs_paths,
        preprocessing=preprocessing, read_image_type=read_image_type)

    if validate:
        val_gen = image_segmentation_generator(
            val_images, val_annotations,  val_batch_size,
            n_classes, input_height, input_width, output_height, output_width,
            other_inputs_paths=other_inputs_paths,
            preprocessing=preprocessing, read_image_type=read_image_type)

    if callbacks is None and (not checkpoints_path is  None) :
        default_callback = ModelCheckpoint(
                filepath=checkpoints_path + ".{epoch:05d}",
                save_weights_only=True,
                verbose=True
            )

        if sys.version_info[0] < 3: # for pyhton 2 
            default_callback = CheckpointsCallback(checkpoints_path)

        callbacks = [
            default_callback
        ]

    if callbacks is None:
        callbacks = []

    if not validate:
        model.fit(train_gen, steps_per_epoch=steps_per_epoch,
                  epochs=epochs, callbacks=callbacks, initial_epoch=initial_epoch)
    else:
        model.fit(train_gen,
                  steps_per_epoch=steps_per_epoch,
                  validation_data=val_gen,
                  validation_steps=val_steps_per_epoch,
                  epochs=epochs, callbacks=callbacks, 
                  initial_epoch=initial_epoch)
        
    return model

def handle_model(train_images, train_anno, val_images, val_anno, input, model, augment, no_validate, classes, verify, epochs, batch_size, steps, zero_ignore):
    callbacks = [
        #keras.callbacks.EarlyStopping(patience=5),
        keras.callbacks.ModelCheckpoint(filepath='checkpoints/model_##name##.{epoch:02d}_##data##.keras'.replace("##name##", model).replace('##data##', os.path.basename(os.path.normpath(input)))),
        #keras.callbacks.ModelCheckpoint(filepath='checkpoints/model_##name##.{epoch:02d}_##data##.ckpt'.replace("##name##", model).replace('##data##', os.path.basename(os.path.normpath(input)))),
        keras.callbacks.TensorBoard(log_dir='./logs'),
        keras.callbacks.ModelCheckpoint(filepath='out/best_##name##_##data##.keras'.replace('##name##', model).replace('##data##', os.path.basename(os.path.normpath(input))), save_best_only=True, mode='max', monitor='val_mean_io_u')
    ]

    trained = train(model, 
          train_images, 
          train_anno,
          validate=not no_validate,
          val_images=val_images, 
          val_annotations=val_anno, 
          verify_dataset=verify,
          callbacks=callbacks, 
          epochs=epochs,
          do_augment=augment,
          n_classes=classes,
          batch_size=batch_size,
          steps_per_epoch=steps,
          ignore_zero_class=zero_ignore)
    
    trained.save('out/best_##name##_##data##.keras'.replace('##name##', model).replace('##data##', os.path.basename(os.path.normpath(input))))
        

@click.command()
@click.option('-m', '--model', type=str)
@click.option('-i', '--input', type=str)
@click.option('-a', '--augment', is_flag=True)
@click.option('-n', '--no-validate', is_flag=True)
@click.option('-c', '--classes', default=25)
@click.option('-v', '--verify', is_flag=True)
@click.option('-e', '--epochs', type=int, default=25)
@click.option('-b', '--batch-size', type=int, default=8)
@click.option('-s', '--steps', type=int, default=512)
@click.option('-z', '--zero-ignore', is_flag=True)
def main(model, input, augment, no_validate, classes, verify, epochs, batch_size, steps, zero_ignore):
    train_images = os.path.join(input, "train", "images")
    train_anno = os.path.join(input, "train", "leaf_instances")

    val_images = os.path.join(input, "val", "images")
    val_anno = os.path.join(input, "val", "leaf_instances")

    if model is None:
        model = list(all_models.model_from_name.keys())
    elif type(model) is str:
        if ',' in model:
            model = model.split(',')
    else:
        raise Error("Please provide model strings")

    if type(model) is not list:
        model = [model]
    for m in model:
        handle_model(train_images=train_images,
                train_anno=train_anno,
                val_images=val_images,
                val_anno=val_anno,
                input=input,
                model=m,
                augment=augment,
                no_validate=no_validate,
                classes=classes,
                verify=verify,
                epochs=epochs,
                batch_size=batch_size,
                steps=steps,
                zero_ignore=zero_ignore)
    
if __name__ == '__main__':
    main()