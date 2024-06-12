import torch
import cv2
import os

class LeafMaskDataset(torch.utils.data.Dataset):
    """Some Information LeafMaskDataset"""
    def __init__(self, data_path, transforms, fields):
        super(LeafMaskDataset, self).__init__()
        self.data_path = data_path
        self.transforms = transforms
        self.fields = fields
        self.image_list = os.listdir(os.path.join(data_path, "images"))

    def __getitem__(self, index):
        img_name = self.image_list[index]
        image = cv2.imread(os.path.join(self.data_path, "images", img_name))
        annotations = dict([(field, cv2.imread(os.path.join(self.data_path, field, img_name))) for field in self.fields])
        return self.transforms(image, annotations) if self.transforms else (image, annotations)

    def __len__(self):
        return len(self.image_list)