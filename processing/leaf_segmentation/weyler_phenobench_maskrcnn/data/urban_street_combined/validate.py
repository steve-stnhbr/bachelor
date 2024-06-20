import os
import cv2
import numpy as np
import colorsys

HUE_STEP = 15

VERBOSE = False

IMAGE_FOLDER = "images"
MASK_FOLDER = "plant_instances"
INSTANCE_FOLDER = "leaf_instances"

def main():
    for file in os.listdir(IMAGE_FOLDER):
        if not os.path.isfile(os.path.join(IMAGE_FOLDER, file)):
            continue
        
        image = cv2.imread(os.path.join(IMAGE_FOLDER, file))
        mask = cv2.imread(os.path.join(MASK_FOLDER, file), cv2.IMREAD_GRAYSCALE)
        instances = cv2.imread(os.path.join(INSTANCE_FOLDER, file), cv2.IMREAD_GRAYSCALE)

        bboxes = calculate_boxes(instances)
        if VERBOSE:
            print(bboxes)
        show_image(image, mask, bboxes)


def calculate_boxes(instances):
    instance_values = np.unique(instances)
    index = np.argwhere(instance_values == 0)
    instance_values = np.delete(instance_values, index)
    boxes = []

    for value in instance_values:
        mask = np.where(instances == value)
        boxes.append(calculate_bbox(mask))
    return boxes

def calculate_bbox(mask):
    if len(mask) != 0 and len(mask[1]) != 0 and len(mask[0]) != 0:
        x_min = int(np.min(mask[1]))
        x_max = int(np.max(mask[1]))
        y_min = int(np.min(mask[0]))
        y_max = int(np.max(mask[0]))

        return x_min, x_max, y_min, y_max
    else:
        return 0, 0, 0, 0 
        

def show_image(image, mask, boxes, bbox=True, overlay=True):
    cv2_image_bbox = image.copy()
    if overlay:
        cv2_image_bbox //= 2
        if mask.shape == image.shape[:2]:
            cv2_image_bbox[mask > 0] *= 2
        else:
            print("Could not show instances")
    if bbox:
        for j, box in enumerate(boxes):
            top_left = (box[0], box[2])
            bottom_right = (box[1], box[3])
            if box[0] - box[1] == 0 or box[2] - box[3] == 0:
                continue
            cv2.rectangle(cv2_image_bbox, top_left, bottom_right, [el * 255 for el in colorsys.hsv_to_rgb((j * HUE_STEP) / 255, 1, 1)], 2)
    if VERBOSE:
        print(image.shape, mask.shape)
    
    cv2.imshow("image", cv2_image_bbox)
    cv2.waitKey(0)

if __name__ == '__main__': 
    main()