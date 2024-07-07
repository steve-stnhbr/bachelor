import numpy as np
from keras.utils import Sequence
import os
from cv2 import imread, resize
import lib.Mask_RCNN.mrcnn.utils as utils
import cv2

class CustomMRCNNDataset(Sequence):
    def __init__(self, image_dir, mask_dir, batch_size, image_size, config):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.config = config
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)
    
    def __len__(self):
        return int(np.ceil(len(self.images) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_images = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_masks = self.masks[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        image_batch = []
        mask_batch = []
        image_meta_batch = []
        anchors_batch = []

        for image_name, mask_name in zip(batch_images, batch_masks):
            image = imread(os.path.join(self.image_dir, image_name))
            mask = imread(os.path.join(self.mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
            
            # Convert mask to binary (0 and 1 values)
            mask = (mask > 0).astype(np.uint8)
            
            # Resize image and mask
            image = resize(image, self.image_size)
            mask = resize(mask, self.image_size)

            # Generate image meta data and anchors
            image_meta = compose_image_meta(
                0, image.shape, image.shape, np.zeros([self.config.NUM_CLASSES], dtype=np.int32)
            )
            anchors = utils.generate_pyramid_anchors(
                self.config.RPN_ANCHOR_SCALES,
                self.config.RPN_ANCHOR_RATIOS,
                self.config.BACKBONE_SHAPES,
                self.config.BACKBONE_STRIDES,
                self.config.RPN_ANCHOR_STRIDE
            )

            image_batch.append(image)
            mask_batch.append(mask)
            image_meta_batch.append(image_meta)
            anchors_batch.append(anchors)
        
        return (np.array(image_batch), np.array(image_meta_batch), np.array(anchors_batch)), np.array(mask_batch)

# def compose_image_meta(image_id, original_image_shape, image_shape, active_class_ids):
#     """Builds an array of image attributes."""
#     meta = np.array(
#         [image_id] + 
#         list(original_image_shape) + 
#         list(image_shape) + 
#         list(active_class_ids)
#     )
#     return meta
def compose_image_meta(image_id, original_image_shape, image_shape, window, scale, active_class_ids):
    """Builds an array of image attributes."""
    meta = np.array(
        [image_id] + 
        list(original_image_shape) + 
        list(image_shape) + 
        list(window) + 
        [scale] + 
        list(active_class_ids)
    )
    return meta
