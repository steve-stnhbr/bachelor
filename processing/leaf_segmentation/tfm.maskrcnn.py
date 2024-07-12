import tensorflow as tf
import numpy as np
from PIL import Image
import os
import click
import tensorflow as tf
from official.vision.models import MaskRCNNModel
from official.vision.configs import MaskRCNN as MaskRCNNConfig

CLASSES = 1

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

# Define your configuration
class CustomMaskRCNNConfig(MaskRCNNConfig):
    def __init__(self):
        super().__init__()
        self.num_classes = 3  # Adjust this based on your number of classes
        self.image_size = [224, 224, 3]  # Adjust based on your image size
        # Add other custom configurations as needed

@click.command()
@click.argument('input_path')
@click.option('-e', '--epochs', type=int, default=10)
def main(input_path, epochs):
    # Create the model
    config = CustomMaskRCNNConfig()
    model = MaskRCNNModel(config)

    train_dir = os.path.join(input_path, "train")
    train_images_dir = os.path.join(train_dir, 'images')
    train_masks_dir = os.path.join(train_dir, 'leaf_instances')

    val_dir = os.path.join(input_path, "val")
    val_images_dir = os.path.join(val_dir, 'images')
    val_masks_dir = os.path.join(val_dir, 'leaf_instances')

    class_ids = range(CLASSES)

    # Prepare your dataset
    train_dataset = CustomMaskRCNNDataset(
        image_dir=train_images_dir,
        mask_dir=train_masks_dir,
        class_ids=class_ids,  # Adjust based on your class IDs
        image_size=(224, 224),
        batch_size=2
    )

    val_dataset = CustomMaskRCNNDataset(
        image_dir=val_images_dir,
        mask_dir=val_masks_dir,
        class_ids=class_ids,  # Adjust based on your class IDs
        image_size=(224, 224),
        batch_size=2
    )

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer)

    # Train the model
    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset
    )

    # Save the model
    model.save('mask_rcnn_model_weights.keras')

    # Inference
    def predict(model, image_path):
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        
        predictions = model.predict(image)
        # Process predictions as needed
        return predictions

    # Example usage
    test_image_path = 'path/to/test/image.png'
    results = predict(model, test_image_path)