import tensorflow as tf
import tensorflow_models as tfm

from official.vision.configs import maskrcnn as exp_cfg
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.vision.tasks import maskrcnn
from official.vision.dataloaders.tf_example_decoder import TfExampleDecoder
from official.vision.serving import export_saved_model_lib
from official.vision.configs import backbones as backbones_cfg
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
import numpy as np
from official.vision.utils.object_detection import visualization_utils
import time
from custom_utils import send_pushover_notification, intercept_stdout

IMAGE_SIZE = (640, 640)
BATCH_SIZE = 4
TFDS_NAME = 'leaf_instance_dataset'
INPUT_PATH = "/home/stefan.steinheber/tensorflow_datasets/leaf_instance_dataset/1.0.0/"
MODEL = "bbii"

def build_experiment_config():
    # Create a base experiment config
    exp_config = exp_factory.get_exp_config('maskrcnn_resnetfpn_coco')

    # exp_config.task.model.backbone = backbones_cfg.Backbone(
    #     type='vit',
    #     vit=backbones_cfg.VisionTransformer(
    #         model_name='vit-b16',
    #         representation_size=768,
    #         patch_size=16,
    #         hidden_size=768,
    #         output_2d_feature_maps=True,
    #         transformer=backbones_cfg.Transformer(
    #             num_layers=12,
    #             mlp_dim=3072,
    #             num_heads=12,
    #             dropout_rate=0.1,
    #             attention_dropout_rate=0.1
    #         )
    #     )
    # )
    # exp_config.task.model.backbone.output_shape = (14, 14, 768) 
    # exp_config.task.model.backbone.vit.output_shape = (14, 14, 768) 

    exp_config.task.model.input_size = [IMAGE_SIZE[1], IMAGE_SIZE[0], 3]

    # Modify the config as needed
    exp_config.task.model.num_classes = 2  # Adjust based on your number of classes
    # exp_config.task.model.mask_head.num_convs = 4
    # exp_config.task.model.mask_head.num_filters = 256
    # exp_config.task.model.mask_head.use_separable_conv = False
    
    # Configure for custom dataset
    exp_config.task.train_data.input_path = INPUT_PATH + "*train*"
    exp_config.task.validation_data.input_path = INPUT_PATH + "*val*"
    exp_config.task.train_data.global_batch_size = BATCH_SIZE
    exp_config.task.validation_data.global_batch_size = BATCH_SIZE

    # Disable COCO-specific configurations
    exp_config.task.annotation_file = INPUT_PATH + "instances.json"
    exp_config.task.use_coco_metrics = False

    # Configure data parsers
    exp_config.task.train_data.parser = exp_cfg.Parser()
    exp_config.task.validation_data.parser = exp_cfg.Parser()

    # Training parameters
    train_steps = 224_000
    exp_config.trainer.steps_per_loop = 200
    exp_config.trainer.summary_interval = 200
    exp_config.trainer.checkpoint_interval = 200
    exp_config.trainer.validation_interval = 200
    exp_config.trainer.validation_steps = 200
    exp_config.trainer.train_steps = train_steps
    exp_config.trainer.optimizer_config.warmup.linear.warmup_steps = 200
    exp_config.trainer.optimizer_config.learning_rate.type = 'cosine'
    exp_config.trainer.optimizer_config.learning_rate.cosine.decay_steps = train_steps
    exp_config.trainer.optimizer_config.learning_rate.cosine.initial_learning_rate = 0.07
    exp_config.trainer.optimizer_config.warmup.linear.warmup_learning_rate = 0.05

    return exp_config

# dataset = tfds.load('leaf_instance_dataset', split='train')
# for example in dataset.take(1):
#     print(example)

# Build the task with your custom dataset
exp_config = build_experiment_config()

logical_device_names = [logical_device.name for logical_device in tf.config.list_logical_devices('GPU')]

if len(logical_device_names) == 0:
    logical_device_names = [logical_device.name for logical_device in tf.config.list_logical_devices()]

distribution_strategy = tf.distribute.OneDeviceStrategy(logical_device_names[0])

print("Created distribution Strategy on Device", logical_device_names[0])

with distribution_strategy.scope():
    model_dir = "out/" + MODEL
    task = tfm.core.task_factory.get_task(exp_config.task, logging_dir=model_dir)

def show_batch(raw_records):
    tf_ex_decoder = TfExampleDecoder(include_mask=True)
    plt.figure(figsize=(20, 20))
    use_normalized_coordinates=True
    min_score_thresh = 0.30
    for i, serialized_example in enumerate(raw_records):
        plt.subplot(1, 3, i + 1)
        decoded_tensors = tf_ex_decoder.decode(serialized_example)
        image = decoded_tensors['image'].numpy().astype('uint8')
        scores = np.ones(shape=(len(decoded_tensors['groundtruth_boxes'])))
        # print(decoded_tensors['groundtruth_instance_masks'].numpy().shape)
        # print(decoded_tensors.keys())
        visualization_utils.visualize_boxes_and_labels_on_image_array(
            image,
            decoded_tensors['groundtruth_boxes'].numpy(),
            decoded_tensors['groundtruth_classes'].numpy().astype('int'),
            scores,
            category_index={
                1: {
                    'id': 1,
                    'name': 'leaf',
                },
            },
            use_normalized_coordinates=use_normalized_coordinates,
            min_score_thresh=min_score_thresh,
            instance_masks=decoded_tensors['groundtruth_instance_masks'].numpy().astype('uint8'),
            line_thickness=4)

        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Image-{i+1}")
    #plt.show()
    plt.savefig("out/fig.png")

buffer_size = 100
num_of_examples = 3

train_tfrecords = tf.io.gfile.glob(exp_config.task.train_data.input_path)
raw_records = tf.data.TFRecordDataset(train_tfrecords).shuffle(buffer_size=buffer_size).take(num_of_examples)
show_batch(raw_records)

send_pushover_notification("Starting Training", "Tensorflow Models")


model, eval_logs = tfm.core.train_lib.run_experiment(
    distribution_strategy=distribution_strategy,
    task=task,
    mode='train',
    params=exp_config,
    model_dir=model_dir,
    run_post_eval=False)

send_pushover_notification("Finished Training", "Tensorflow Models")


# tf.keras.utils.plot_model(model, show_shapes=True)

# for key, value in eval_logs.items():
#     if isinstance(value, tf.Tensor):
#         value = value.numpy()
#     print(f'{key:20}: {value:.3f}')

export_saved_model_lib.export_inference_graph(
    input_type='image_tensor',
    batch_size=1,
    input_image_size=[IMAGE_SIZE[1], IMAGE_SIZE[0]],
    params=exp_config,
    checkpoint_path=tf.train.latest_checkpoint(model_dir),
    export_dir=f'out/mask_rcnn_{time.time()}')