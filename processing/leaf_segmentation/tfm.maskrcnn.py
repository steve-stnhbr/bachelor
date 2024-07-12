import tensorflow as tf
import tensorflow_datasets as tfds
from official.vision.configs import maskrcnn as exp_cfg
from official.vision.configs import common as common_cfg
from official.vision.modeling import factory
from official.core import exp_factory
from official.core import config_definitions as cfg
from official.vision.serving import export_saved_model_lib


experiment_type = 'maskrcnn_resnetfpn_coco'
config = exp_factory.get_exp_config(experiment_type)

# Modify the configuration as needed
config.task.model.num_classes = 91  # Number of COCO classes
config.task.model.mask_head.num_convs = 4
config.task.model.mask_head.num_filters = 256
config.task.model.roi_sampler.num_samples = 512
config.task.losses.l2_weight_decay = 0.00004
config.task.train_data.global_batch_size = 16
config.task.validation_data.global_batch_size = 16
config.trainer.train_steps = 10000
config.trainer.validation_steps = 100
config.trainer.validation_interval = 1000
config.trainer.checkpoint_interval = 1000
config.trainer.optimizer_config.learning_rate.type = 'stepwise'
config.trainer.optimizer_config.learning_rate.stepwise.boundaries = [6000, 8000]
config.trainer.optimizer_config.learning_rate.stepwise.values = [0.08, 0.008, 0.0008]

def preprocess_example(example):
    image = example['image']
    boxes = example['objects']['bbox']
    classes = example['objects']['label']
    masks = example['objects']['mask']
    
    image = tf.image.resize(image, [640, 640])
    image = tf.keras.applications.resnet.preprocess_input(image)
    
    return {
        'image': image,
        'groundtruth_boxes': boxes,
        'groundtruth_classes': classes,
        'groundtruth_masks': masks
    }



train_dataset = tfds.load('coco/2017')
train_dataset = train_dataset.batch(config.task.train_data.global_batch_size)

model_builder = factory.build_maskrcnn
model = model_builder(config.task.model)

model.fit(train_dataset)