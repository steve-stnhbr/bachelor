"""leaf_coco dataset."""

import tensorflow_datasets as tfds
import os
import tensorflow as tf
import numpy as np
from PIL import Image

def masks_to_boxes(masks, area_threshold=50):
    # if masks.numel() == 0:
    #     return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    n = masks.shape[0]
    width = masks.shape[1]
    height = masks.shape[2]

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
    bounding_boxes[:, 0] /= width
    bounding_boxes[:, 1] /= height
    bounding_boxes[:, 2] /= width
    bounding_boxes[:, 3] /= height
    return bounding_boxes#, bounding_boxes_area

def class_labels_to_masks(labels):
# Get the shape of the input array
    x, y, _ = labels.shape

    # Find unique values in the n dimension
    unique_values = np.unique(labels)

    # Number of unique values
    u = len(unique_values)

    # Initialize the new array with zeros
    masks = np.zeros((u, x, y), dtype=int)

    # Create the binary mask for each unique value
    for i, val in enumerate(unique_values):
        masks[i] = np.any(mask == val, axis=2).astype(int)

    return masks

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
            features=tfds.features.FeaturesDict({
                'image': tfds.features.Image(shape=(None, None, 3), dtype=tf.uint8),
                'image/filename': tfds.features.Text(),
                'image/id': tf.int64,
                'objects': tfds.features.Sequence({
                    'area': tf.float16,
                    'bbox': tfds.features.BBoxFeature(),
                    'id': tf.int64,
                    'is_crowd': tf.bool,
                    'label': tfds.features.ClassLabel(num_classes=80),
                }),
            }),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # Specify the path to your dataset
        path = 'data'
        return {
            'train': self._generate_examples(os.path.join(path, 'train')),
            'val': self._generate_examples(os.path.join(path, 'val')),
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
                labels = np.array(Image.open(mask_path))
                
                # Ensure labels is 2D (H, W) and convert to 3D (H, W, 1)
                if labels.ndim == 2:
                    labels = labels[..., np.newaxis]
                    
                masks = class_labels_to_masks(labels)

                example = {
                    'image': image,
                    'image/filename': filename,
                    'image/id': i,
                    'objects': [
                        {
                            'id': i * j,
                            'bbox': bbox,
                            'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                            'is_crowd': False,
                            'label': 1
                        }
                        for j, bbox in enumerate(masks_to_boxes(masks))
                    ]
                }
                
                yield i, example