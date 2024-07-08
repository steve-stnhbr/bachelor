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
    img_path, mask_path, image_id, annotation_id_start = args
    mask_image = Image.open(mask_path)
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
        
        annotation = create_coco_annotation(category_mask, image_id, int(category_id), annotation_id)
        annotations.append(annotation)
        annotation_id += 1
    
    image_info = {
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": os.path.basename(img_path)
    }
    
    return image_info, annotations, list(unique_categories)

def convert_masks_to_coco(image_dir, mask_dir, output_path):
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    
    images = []
    annotations = []
    categories = set()
    
    with Pool(cpu_count()) as pool:
        image_id = 1
        annotation_id_start = 1
        args = [(img_path, mask_path, image_id + idx, annotation_id_start + idx * 1000) 
                for idx, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths))]
        
        results = list(tqdm.tqdm(pool.map(process_single_image, args), total = len(image_paths)))
        
        for image_info, annots, cats in results:
            images.append(image_info)
            annotations.extend(annots)
            categories.update(cats)
    
    categories = [{"id": int(category_id), "name": f"category_{category_id}"} for category_id in sorted(categories)]
    
    coco_dataset = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    with open(output_path, 'w') as f:
        json.dump(coco_dataset, f, indent=4)


@click.command()
@click.option("-i", '--images', type=str)
@click.option("-m", '--masks', type=str)
@click.option("-o", '--output', type=str)
def main(images, masks, output):
    convert_masks_to_coco(images, masks, output)

if __name__ == "__main__":
    main()
    
