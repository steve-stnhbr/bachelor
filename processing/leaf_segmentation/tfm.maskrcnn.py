import tensorflow as tf
import tensorflow_datasets as tfds
from official.vision.configs import maskrcnn as exp_cfg
from official.vision.configs import common as common_cfg
from official.vision.modeling import factory
from official.core import exp_factory
from official.core import config_definitions as cfg
from official.vision.serving import export_saved_model_lib
import os

import numpy as np


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

def masks_to_boxes(masks, area_threshold=50):
    # if masks.numel() == 0:
    #     return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    n = masks.shape[0]

    bounding_boxes = np.zeros(
        (n, 4), dtype=np.float16)

    for index, mask in enumerate(masks):
        if mask.sum() < area_threshold:
            continue
        y, x = np.nonzero(mask)
        bounding_boxes[index, 0] = np.min(x)
        bounding_boxes[index, 1] = np.min(y)
        bounding_boxes[index, 2] = np.max(x)
        bounding_boxes[index, 3] = np.max(y)
    bounding_boxes_area = bounding_boxes.sum(axis=1)
    bounding_boxes = bounding_boxes[~(bounding_boxes_area==0)]
    return bounding_boxes, bounding_boxes_area

def _load_data(image_path, mask_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    
    filename = tf.strings.split(image_path, os.path.sep)[-1]
    image_id = tf.strings.to_number(tf.strings.split(filename, '.')[0], out_type=tf.int32)
    
    # You'll need to implement masks_to_boxes function
    boxes, areas = tf.py_function(self.masks_to_boxes, [mask], [tf.float32, tf.float32])
    
    return {
        'image': image,
        'image/filename': filename,
        'image/id': image_id,
        'objects': {
            'id': tf.range(tf.shape(boxes)[0], dtype=tf.int32),
            'bbox': boxes,
            'area': areas,
            'is_crowd': tf.zeros(tf.shape(boxes)[0], dtype=tf.bool),
            'label': tf.ones(tf.shape(boxes)[0], dtype=tf.int32)
        }
    }

class LeafInstanceDataset(tf.data.Dataset):
    def __init__(self, path, mask_path='leaf_instance'):
        super().__init__()
        self.path = path
        self.image_files = [os.path.join(path, 'images', file) for file in os.listdir(os.path.join(path, 'images'))]
        self.mask_files = [os.path.join(path, mask_path, file) for file in os.listdir(os.path.join(path, mask_path))]

        if len(self.image_files) != len(self.mask_files):
            raise Error("Dirs different files")

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx]
        
        # Load image and mask
        image = np.array(Image.open(image_path))
        mask = np.array(Image.open(mask_path))
        
        # Ensure mask is 2D (H, W) and convert to 3D (H, W, 1)
        if mask.ndim == 2:
            mask = mask[..., np.newaxis]

        yield {
            'image': image,
            'image/filename': filename,
            'image/id': i,
            'objects': [
                {
                    'id': i * j,
                    'bbox': bbox,
                    'area': area,
                    'is_crowd': False,
                    'label': 1
                }
                for i, (bbox, area) in enumerate(masks_to_boxes(mask))
            ]
        }


path = "_data/combined/train"
mask_path = "leaf_instances"

image_files = [os.path.join(path, 'images', file) for file in os.listdir(os.path.join(path, 'images'))]
image_files = tf.convert_to_tensor(image_files, dtype=tf.string)

mask_files = [os.path.join(path, mask_path, file) for file in os.listdir(os.path.join(path, mask_path))]
mask_files = tf.convert_to_tensor(mask_files, dtype=tf.string)

train_dataset = tf.data.Dataset.from_tensor_slices((image_files, mask_files))
train_dataset = train_dataset.map(_load_data, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(config.task.train_data.global_batch_size)

model_builder = factory.build_maskrcnn
model = model_builder(config.task.model)

model.fit(train_dataset)