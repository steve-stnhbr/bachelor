import tensorflow as tf
import numpy as np
from PIL import Image
import os

class CustomMaskRCNNDataset(tf.keras.utils.Sequence):
    def __init__(self, image_dir, mask_dir, class_ids, image_size=(224, 224), batch_size=1):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.class_ids = class_ids
        self.image_size = image_size
        self.batch_size = batch_size
        
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        
    def __len__(self):
        return len(self.image_files) // self.batch_size
    
    def __getitem__(self, idx):
        batch_image_files = self.image_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_mask_files = self.mask_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        images = []
        masks = []
        
        for image_file, mask_file in zip(batch_image_files, batch_mask_files):
            # Load and preprocess image
            image = Image.open(os.path.join(self.image_dir, image_file)).convert('RGB')
            image = image.resize(self.image_size)
            image = np.array(image) / 255.0
            
            # Load and preprocess mask
            mask = Image.open(os.path.join(self.mask_dir, mask_file)).convert('L')
            mask = mask.resize(self.image_size)
            mask = np.array(mask)
            
            # Convert mask to binary masks for each class
            binary_masks = []
            for class_id in self.class_ids:
                binary_mask = (mask == class_id).astype(np.uint8)
                binary_masks.append(binary_mask)
            
            images.append(image)
            masks.append(np.stack(binary_masks, axis=-1))
        
        return np.array(images), np.array(masks)

class CustomMaskRCNNDataset(tf.keras.utils.Sequence):
    def __init__(self, image_dir, mask_dir, class_ids, image_size=(224, 224), batch_size=1):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.class_ids = class_ids
        self.image_size = image_size
        self.batch_size = batch_size
        
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        
    def __len__(self):
        return len(self.image_files) // self.batch_size
    
    def __getitem__(self, idx):
        batch_image_files = self.image_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_mask_files = self.mask_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        images = []
        masks = []
        
        for image_file, mask_file in zip(batch_image_files, batch_mask_files):
            # Load and preprocess image
            image = Image.open(os.path.join(self.image_dir, image_file)).convert('RGB')
            image = image.resize(self.image_size)
            image = np.array(image) / 255.0
            
            # Load and preprocess mask
            mask = Image.open(os.path.join(self.mask_dir, mask_file)).convert('L')
            mask = mask.resize(self.image_size)
            mask = np.array(mask)
            
            # Convert mask to binary masks for each class
            binary_masks = []
            for class_id in self.class_ids:
                binary_mask = (mask == class_id).astype(np.uint8)
                binary_masks.append(binary_mask)
            
            images.append(image)
            masks.append(np.stack(binary_masks, axis=-1))
        
        return np.array(images), np.array(masks)