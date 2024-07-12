import tensorflow as tf
import tensorflow_datasets as tfds
from official.vision.configs import maskrcnn as exp_cfg
from official.vision.configs import common as common_cfg
from official.vision.modeling import factory
from official.core import exp_factory
from official.core import config_definitions as cfg
from official.vision.serving import export_saved_model_lib

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

def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
        """
        Compute the bounding boxes around the provided masks.

        Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
        ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

        Args:
            masks (Tensor[N, H, W]): masks to transform where N is the number of masks
                and (H, W) are the spatial dimensions.

        Returns:
            Tensor[N, 4]: bounding boxes
        """
        # if masks.numel() == 0:
        #     return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

        n = masks.shape[0]

        bounding_boxes = np.zeros(
            (n, 4), device=masks.device, dtype=torch.float)

        for index, mask in enumerate(masks):
            if mask.sum() < self.area_threshold:
                continue
            y, x = np.nonzero(mask)
            bounding_boxes[index, 0] = np.min(x)
            bounding_boxes[index, 1] = np.min(y)
            bounding_boxes[index, 2] = np.max(x)
            bounding_boxes[index, 3] = np.max(y)
        bounding_boxes_area = bounding_boxes.sum(dim=1)
        bounding_boxes = bounding_boxes[~(bounding_boxes_area==0)]
        return bounding_boxes, bounding_boxes_area

class LeafInstanceDataset(tfds.core.GeneratorBasedBuilder):
    """Leaf Instance dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description="Leaf instance dataset with RGB images and instance masks.",
            features=FeaturesDict({
                'image': tfds.features.Image(shape=(None, None, 3), dtype=np.uint8),
                'image/filename': tfds.features.Text(shape=(), dtype=str),
                'image/id': np.int64,
                'objects': tfds.features.Sequence({
                    'area': np.int64,
                    'bbox': tfds.features.BBoxFeature(shape=(4,), dtype=float32),
                    'id': np.int64,
                    'is_crowd': bool,
                    'label': tfds.features.ClassLabel(shape=(), dtype=int64, num_classes=80),
                }),
            }),
            supervised_keys=('image', 'segmentation_mask'),
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # Specify the path to your dataset
        path = '_data/combined'
        return {
            'train': self._generate_examples(os.path.join(path, 'train')),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        images_dir = os.path.join(path, 'images')
        masks_dir = os.path.join(path, 'leaf_instances')
        
        for i, filename in enumerate(os.listdir(images_dir)):
            if filename.endswith('.png'):
                image_path = os.path.join(images_dir, filename)
                mask_path = os.path.join(masks_dir, filename)
                
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
                        for i, bbox, area in enumerate(masks_to_boxes(mask))
                    ]
                }

tfds.load.register_dataset_builder('leaf_detection', LeafInstanceDataset)
train_dataset = tfds.load('leaf_detection', split='train')
train_dataset = train_dataset.batch(config.task.train_data.global_batch_size)

model_builder = factory.build_maskrcnn
model = model_builder(config.task.model)

model.fit(train_dataset)