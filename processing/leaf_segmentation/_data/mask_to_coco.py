import numpy as np
import cv2
from PIL import Image
from pycocotools import mask as maskUtils
import json
import os
import glob
import click
import tqdm
from multiprocessing import Pool, cpu_count

VERBOSE = False
DEFAULT_COCO_CATEGORIES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'] 

def create_coco_annotation(mask, image_id, category_id, annotation_id):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentations = []
    for contour in contours:
        contour = contour.flatten().tolist()
        segmentations.append(contour)
    
    rle = maskUtils.encode(np.asfortranarray(mask))
    area = maskUtils.area(rle)
    bbox = maskUtils.toBbox(rle)
    
    annotation = {
        "segmentation": segmentations,
        "area": float(area),
        "iscrowd": 0,
        "image_id": image_id,
        "bbox": bbox.tolist(),
        "category_id": category_id,
        "id": annotation_id
    }
    return annotation

def process_single_image(args):
    img_path, mask_path, image_id, annotation_id_start, fixed_category_id = args
    if VERBOSE:
        print(f"Converting {os.path.basename(img_path)}")
    mask_image = Image.open(mask_path).convert('L')
    mask = np.array(mask_image)
    
    unique_categories = np.unique(mask)
    unique_categories = unique_categories[unique_categories > 0]  # Exclude background (assumed to be 0)
    
    annotations = []
    height, width = mask.shape
    annotation_id = annotation_id_start
    
    for category_id in unique_categories:
        category_mask = (mask == category_id).astype(np.uint8)
        if category_mask.sum() == 0:
            continue
        
        annotation = create_coco_annotation(category_mask, image_id, int(category_id) if fixed_category_id is None else fixed_category_id, annotation_id)
        annotations.append(annotation)
        annotation_id += 1
    
    image_info = {
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": os.path.basename(img_path)
    }
    
    return image_info, annotations, list(unique_categories) if fixed_category_id is None else [fixed_category_id]

def convert_masks_to_coco(image_dir, mask_dir, output_path, pool_size=None, category=None, default_categories=False):
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    
    images = []
    annotations = []
    categories = set()

    if pool_size is None:
        pool_size = int(cpu_count() * .8)
    print(f"Spawning pool with {pool_size} workers")
    
    with Pool(pool_size) as pool:
        image_id = 1
        annotation_id_start = 1
        args = [(img_path, mask_path, image_id + idx, annotation_id_start + idx * 1000, None if category is None else category[0]) 
                for idx, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths))]
        
        print("Starting Pool")
        
        results = list(tqdm.tqdm(pool.imap(process_single_image, args), total = len(image_paths)))
        
        for image_info, annots, cats in results:
            images.append(image_info)
            annotations.extend(annots)
            categories.update(cats)
    
    if default_categories:
        categories = [
            {
                'id': index,
                'name': name
            }
            for index, name in enumerate(DEFAULT_COCO_CATEGORIES)
        ]
    elif category is None:
        categories = [
            {
                "id": int(category_id), 
                "name": f"category_{category_id}"
            }
            for category_id in set(sorted(categories))
        ]
    else:
        categories = [
            {
                "id": category[0],
                'name': category[1]
            }
        ]
    
    coco_dataset = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    print(f"Writing to {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(coco_dataset, f, indent=4)


@click.command()
@click.option("-i", '--images', type=str)
@click.option("-m", '--masks', type=str)
@click.option("-o", '--output', type=str)
@click.option("-p", '--pool_size', type=int, default=None)
@click.option('--fixed-category-id', type=int)
@click.option('--fixed-category-name', type=str)
@click.option('--default-categories', is_flag=True)
def main(images, masks, output, pool_size, fixed_category_id, fixed_category_name, default_categories):
    convert_masks_to_coco(images, masks, output, pool_size=pool_size, category=(fixed_category_id, fixed_category_name), default_categories=default_categories)

if __name__ == "__main__":
    main()
    
