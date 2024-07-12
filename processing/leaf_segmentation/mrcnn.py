import lib.Mask_RCNN.mrcnn.utils as utils
import os
from imgaug import augmenters as iaa
import click
import numpy as np

from lib.Mask_RCNN.mrcnn.config import Config
from lib.Mask_RCNN.mrcnn import utils
from lib.Mask_RCNN.mrcnn import model as modellib
from lib.Mask_RCNN.mrcnn import visualize

CLASS_NAME = "leaf"
CLASSES = 2

class LeavesConfig(Config):
    NAME="mask_rcnn"
    NUM_CLASSES = CLASSES # COCO has 80 classes
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    def compute_backbone_shapes(self, image_shape):
        """Computes the width and height of each stage of the backbone network."""
        return np.array(
            [[int(np.ceil(image_shape[0] / stride)),
            int(np.ceil(image_shape[1] / stride))]
            for stride in self.BACKBONE_STRIDES]
        )

    def __init__(self):
        super().__init__()
        self.BACKBONE_STRIDES = [4, 8, 16, 32, 64]
        self.BACKBONE_SHAPES = self.compute_backbone_shapes([self.IMAGE_MIN_DIM, self.IMAGE_MAX_DIM])


class LeavesDataset(utils.Dataset):
    def __init__(self, files_dir):
        super().__init__()

        self.files_dir = files_dir
        self.load_leaves()

    def load_leaves(self):
        self.add_class(CLASS_NAME, 1, CLASS_NAME)

        for file in self.files_dir:
            file_path = os.path.join(file)

            self.add_image(
                CLASS_NAME,
                image_id=file,
                path=file_path)
            
def train(path, epochs=24, batch_size=8, model_dir=os.getcwd()):
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])

    config = LeavesConfig()

    model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=model_dir)

    dataset_train = LeavesDataset(os.path.join(path, 'train'))
    dataset_val = LeavesDataset(os.path.join(path, 'val'))
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=20,
                augmentation=augmentation,
                layers='heads')
    
@click.command()
@click.argument('file_path')
@click.option('-e', '--epochs', default=24)
@click.option('-b', '--batch_size', default=8)
@click.option('-m', '--model_dir', default=os.getcwd())
def main(file_path, epochs, batch_size, model_dir):
    train(file_path, model_dir=model_dir)



if __name__ == '__main__':
    main()
        