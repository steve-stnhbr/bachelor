import torch
import torch.nn.functional as F
import copy
from dataloaders.pdc_base import PlantsBase 
from torch.utils.data import Dataset
import os
import cv2
import numpy as np

def collate_pdc(items):
    batch = {}
    images = []
    targets = []
    names=[]
    for i in range(len(items)):
        images.append(items[i]['image'])
        targets.append(items[i]['targets'])
        names.append(items[i]['name'])
    batch['image'] = list(images)
    batch['targets'] = list(targets)
    batch['name'] = list(names)
    return batch

class Leaves(PlantsBase):
    def __init__(self, datapath, overfit=False, area_threshold=50, cfg=None, is_train=False):
        super().__init__(datapath, overfit, cfg, is_train)
        self.area_threshold = area_threshold

    def masks_to_boxes(self, masks: torch.Tensor) -> torch.Tensor:
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
        if masks.numel() == 0:
            return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

        n = masks.shape[0]

        bounding_boxes = torch.zeros(
            (n, 4), device=masks.device, dtype=torch.float)

        for index, mask in enumerate(masks):
            if mask.sum() < self.area_threshold :
                continue
            y, x = torch.where(mask != 0)
            bounding_boxes[index, 0] = torch.min(x)
            bounding_boxes[index, 1] = torch.min(y)
            bounding_boxes[index, 2] = torch.max(x)
            bounding_boxes[index, 3] = torch.max(y)
        bounding_boxes_area = bounding_boxes.sum(dim=1)
        bounding_boxes = bounding_boxes[~(bounding_boxes_area==0)]
        return bounding_boxes, bounding_boxes_area


    def __getitem__(self, index):
        sample = self.get_sample(index)
        ignore_mask = sample['ignore_mask']
        instances = sample['leaf_instances'] * (~ignore_mask)
        instances = torch.unique(instances, return_inverse=True)[1]
        semantic = copy.deepcopy(instances)
        semantic[semantic > 0] = 1
        masks = F.one_hot(instances).permute(2, 0, 1) 
        cls_exploded = masks * semantic.unsqueeze(0)
        cls_exploded = torch.reshape(cls_exploded, (cls_exploded.shape[0], cls_exploded.shape[1] * cls_exploded.shape[2]))
        # cls_vec contains the class_id for each masks
        cls_vec, _ = torch.max(cls_exploded, dim=1) 
        # computing bounding boxes from masks
        boxes, area = self.masks_to_boxes(masks)
        # apply reduction for null boxes
        masks = masks[[~(area == 0)]]
        cls_vec = cls_vec[[~(area == 0)]]

        image = (sample['images']/255).float()

        maskrcnn_input = {}
        maskrcnn_input['image'] = image
        maskrcnn_input['name'] = sample['image_name']
        maskrcnn_input['targets'] = {}
        maskrcnn_input['targets']['masks'] = masks.to(torch.uint8)
        maskrcnn_input['targets']['labels'] = cls_vec
        maskrcnn_input['targets']['boxes'] = boxes

        return maskrcnn_input

class LeafMask(Dataset):
    """Some Information about LeafMask"""
    def __init__(self, datapath, area_threshold=50, img_size=(224, 224)):
        super(LeafMask, self).__init__()
        self.datapath = datapath
        self.image_list = [
            x for x in os.listdir(os.path.join(self.datapath, "images")) if ".png" in x
        ]
        
        self.len = len(self.image_list)
        self.area_threshold = area_threshold
        self.img_size = img_size

    def __getitem__(self, index):
        image_name = self.image_list[index]
        image = cv2.imread(
            os.path.join(os.path.join(self.datapath, "images"), image_name),
            cv2.IMREAD_UNCHANGED,
        )
        image = self.add_padding_to_make_img_array_squared(image)
        image = cv2.resize(image, self.img_size)
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = torch.tensor(image).permute(2, 0, 1).float()
        else:
            image = torch.tensor(image.astype("int16"))
            
        mask = self.image_list[index]
        mask = cv2.imread(
            os.path.join(os.path.join(self.datapath, "masks"), mask),
            cv2.IMREAD_UNCHANGED,
        )
        mask = self.add_padding_to_make_img_array_squared(mask)
        mask = cv2.resize(mask, self.img_size)
        mask = np.any(mask > 0, -1)
        if len(mask.shape) > 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            mask = torch.tensor(mask).permute(2, 0, 1)
        else:
            mask = torch.tensor(mask.astype("int16"))

        mask = mask.unsqueeze(0)

        boxes, area = self.masks_to_boxes(mask)
        # apply reduction for null boxes
        mask = mask[[~(area == 0)]]
        
        maskrcnn_input = {}
        maskrcnn_input['image'] = image
        maskrcnn_input['name'] = image_name
        maskrcnn_input['targets'] = {}
        maskrcnn_input['targets']['labels'] = mask.to(torch.int64)
        maskrcnn_input['targets']['masks'] = mask.to(torch.uint8)
        maskrcnn_input['targets']['boxes'] = boxes
        return maskrcnn_input
    
    def masks_to_boxes(self, masks: torch.Tensor) -> torch.Tensor:
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
        if masks.numel() == 0:
            return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

        n = masks.shape[0]

        bounding_boxes = torch.zeros(
            (n, 4), device=masks.device, dtype=torch.float)

        for index, mask in enumerate(masks):
            if mask.sum() < self.area_threshold:
                continue
            y, x = torch.where(mask != 0)
            bounding_boxes[index, 0] = torch.min(x)
            bounding_boxes[index, 1] = torch.min(y)
            bounding_boxes[index, 2] = torch.max(x)
            bounding_boxes[index, 3] = torch.max(y)
        bounding_boxes_area = bounding_boxes.sum(dim=1)
        bounding_boxes = bounding_boxes[~(bounding_boxes_area==0)]
        return bounding_boxes, bounding_boxes_area
    
    def add_padding_to_make_img_array_squared(self, img):
        """ Adds padding to make the image squared.
        # Arguments
            img: an image as an array.
        """
        sizex = img.shape[0]
        sizey = img.shape[1]
        if (sizex == sizey):
            return img
        else:
            maxsize = np.max([sizex, sizey])
            padx = (maxsize - sizex) // 2
            pady = (maxsize - sizey) // 2
            return np.pad(img, pad_width=((padx,padx),(pady,pady),(0,0)))

    def __len__(self):
        return self.len
    
class LeafSegmentationDataset(Dataset):
    def __init__(self, image_path, mask_path, transforms=None, img_size=(224,224), area_threshold=50):
        self.image_paths = [os.path.join(image_path, x) for x in os.listdir(image_path) if ".png" in x]
        self.mask_paths = [os.path.join(mask_path, x) for x in os.listdir(mask_path) if ".png" in x]
        if len(self.image_paths) != len(self.mask_paths):
            raise "Images and Masks have different lengths"
        self.transforms = transforms
        self.img_size = img_size
        self.area_threshold = area_threshold

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.add_padding_to_make_img_array_squared(image)
        image = cv2.resize(image, self.img_size)
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = torch.tensor(image).permute(2, 0, 1).float()
        else:
            image = torch.tensor(image.astype("int16"))

        mask = cv2.imread(self.mask_paths[idx])
        mask = np.array(mask)
        mask = self.add_padding_to_make_img_array_squared(mask)
        mask = cv2.resize(mask, self.img_size)
        mask = np.any(mask > 0, -1)
        # Labels erstellen: Alle Pixel mit Wert 1 als 1 markieren
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:] # Entferne den Hintergrund
        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        boxes, area = self.masks_to_boxes(masks)
        # apply reduction for null boxes
        masks = masks[[~(area == 0)]]

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["boxes"] = boxes

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return {
            "image": image,
            "targets": target,
            "name": self.image_paths[idx]
        }
    
    def masks_to_boxes(self, masks: torch.Tensor) -> torch.Tensor:
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

        bounding_boxes = torch.zeros(
            (n, 4), device=masks.device, dtype=torch.float)

        for index, mask in enumerate(masks):
            if mask.sum() < self.area_threshold:
                continue
            y, x = torch.where(mask != 0)
            bounding_boxes[index, 0] = torch.min(x)
            bounding_boxes[index, 1] = torch.min(y)
            bounding_boxes[index, 2] = torch.max(x)
            bounding_boxes[index, 3] = torch.max(y)
        bounding_boxes_area = bounding_boxes.sum(dim=1)
        bounding_boxes = bounding_boxes[~(bounding_boxes_area==0)]
        return bounding_boxes, bounding_boxes_area
    
    def add_padding_to_make_img_array_squared(self, img):
        """ Adds padding to make the image squared.
        # Arguments
            img: an image as an array.
        """
        sizex = img.shape[0]
        sizey = img.shape[1]
        if (sizex == sizey):
            return img
        else:
            maxsize = np.max([sizex, sizey])
            padx = (maxsize - sizex) // 2
            pady = (maxsize - sizey) // 2
            return np.pad(img, pad_width=((padx,padx),(pady,pady),(0,0)))