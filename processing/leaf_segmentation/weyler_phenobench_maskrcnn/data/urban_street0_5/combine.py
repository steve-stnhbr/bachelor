import cv2
import numpy as np
import os
import random
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

WORKERS = 12
VERBOSE = False
SEMANTIC_VALUE = 3

def apply_mask(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)

def transform_image(image, mask, angle, scale, tx, ty, output_size):
    height, width = output_size
    center = (width // 2, height // 2)

    # Rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    # Translation matrix
    rotation_matrix[0, 2] += tx
    rotation_matrix[1, 2] += ty

    # Apply the transformations to image and mask
    transformed_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    transformed_mask = cv2.warpAffine(mask, rotation_matrix, (width, height))
    return transformed_image, transformed_mask

def create_segmentation_mask(image, mask, value):
    segmented_mask = np.zeros_like(image)
    segmented_mask[mask > 0] = value
    return segmented_mask

def combine_masks(masks):
    combined_mask = np.zeros_like(masks[0])
    for mask in masks:
        combined_mask[mask > 0] = 255
    return combined_mask

def jitter_positions(output_size, element_size, n):
    positions = []
    subpixel_width = output_size[0] // n
    subpixel_height = output_size[1] // n
    
    for i in range(n):
        for j in range(n):
            # Calculate the range for x and y coordinates within the subpixel grid
            x_min = i * subpixel_width
            x_max = (i + 1) * subpixel_width - element_size[0]
            y_min = j * subpixel_height
            y_max = (j + 1) * subpixel_height - element_size[1]
            
            # Ensure x_max and y_max are not less than x_min and y_min
            if x_max < x_min:
                tmp = x_min
                x_min = x_max
                x_max = tmp
            if y_max < y_min:
                tmp = y_min
                y_min = y_max
                y_max = tmp
            
            x = random.randint(x_min, x_max)
            y = random.randint(y_min, y_max)
            
            positions.append((x - output_size[0] // 2, y - output_size[1] // 2))
    random.shuffle(positions)
    
    return positions

def process_images(folder_path, output_folder, image_range, iterations, output_size, n):
    image_folder = os.path.join(folder_path, 'images')
    mask_folder = os.path.join(folder_path, 'leaf_instances')
    background_folder = os.path.join(folder_path, 'backgrounds')
    # Create separate folders for each iteration
    images_output_folder = os.path.join(output_folder, 'images')
    leaf_instances_output_folder = os.path.join(output_folder, 'leaf_instances')
    plant_instances_output_folder = os.path.join(output_folder, 'plant_instances')
    semantics_output_folder = os.path.join(output_folder, "semantics")

    # Get the list of image, mask, and background files
    image_files = sorted([f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))])
    mask_files = sorted([f for f in os.listdir(mask_folder) if os.path.isfile(os.path.join(mask_folder, f))])
    background_files = sorted([f for f in os.listdir(background_folder) if os.path.isfile(os.path.join(background_folder, f))])

    if not background_files:
        raise FileNotFoundError(f"No background images found in {background_folder}")

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(images_output_folder, exist_ok=True)
    os.makedirs(leaf_instances_output_folder, exist_ok=True)
    os.makedirs(plant_instances_output_folder, exist_ok=True)
    os.makedirs(semantics_output_folder, exist_ok=True)

    step = partial(combine, 
                   image_range=image_range,
                   output_size=output_size,
                   n=n,
                   image_files=image_files,
                   background_folder=background_folder,
                   background_files=background_files,
                   image_folder=image_folder,
                   mask_folder=mask_folder,
                   mask_files=mask_files,
                   images_output_folder=images_output_folder,
                   leaf_instances_output_folder=leaf_instances_output_folder,
                   plant_instances_output_folder=plant_instances_output_folder,
                   semantics_output_folder=semantics_output_folder)

    print(f"Spawning thread-pool with {WORKERS} workers")

    with Pool(WORKERS) as p:        
        r = list(tqdm(p.imap(step, range(iterations)), total=iterations))
        
def combine(iteration,
            image_range,
            output_size,
            n,
            image_files, 
            background_folder, 
            background_files, 
            image_folder, 
            mask_folder, 
            mask_files, 
            images_output_folder, 
            leaf_instances_output_folder, 
            plant_instances_output_folder, 
            semantics_output_folder,
            plant_is_leaf=True):
    # Randomly pick the number of images to combine from the provided range
    num_images = random.randint(image_range[0], image_range[1])

    # Randomly select `num_images` from the folder
    selected_indices = random.sample(range(len(image_files)), num_images)

    # Randomly select a background image
    background_image_path = os.path.join(background_folder, random.choice(background_files))
    background_image = cv2.imread(background_image_path)
    if background_image is None:
        raise FileNotFoundError(f"Background image not found at {background_image_path}")
    combined_image = cv2.resize(background_image, output_size)
    segmentation_mask = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)

    # Collect masks for combining into a single mask
    masks_to_combine = []

    # Generate random positions using jitter algorithm
    positions = jitter_positions(output_size, (output_size[0] // n, output_size[1] // n), n)

    for idx, i in enumerate(selected_indices):
        # Read image and corresponding mask
        image_path = os.path.join(image_folder, image_files[i])
        mask_path = os.path.join(mask_folder, mask_files[i])
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Check if the images and masks are loaded properly
        if image is None or mask is None:
            print(f"Error: Image or mask {i+1} not loaded properly")
            continue

        # Resize image and mask to the same size as the original image
        image_resized = cv2.resize(image, (mask.shape[1], mask.shape[0]))

        # Get random position for the transformed element
        if idx < len(positions):
            tx, ty = positions[idx]
        else:
            tx, ty = 0, 0

        # Generate random transformations
        angle = random.uniform(-45, 45)  # Random rotation between -45 and 45 degrees
        scale = random.uniform(0.1, .9)  # Random scaling between 0.5 and 1.5

        # Apply transformations to image and mask
        transformed_image, transformed_mask = transform_image(image_resized, mask, angle, scale, tx, ty, output_size)

        # Create the segmentation mask
        segmentation_value = [idx + 1, idx + 1, idx + 1]
        segmentation_mask[transformed_mask > 0] = segmentation_value

        # Replace pixels in combined_image with transformed_image in the masked region
        combined_image[transformed_mask > 0] = transformed_image[transformed_mask > 0]

        # Collect mask for combining
        masks_to_combine.append(transformed_mask)

    # Combine masks into a single mask (all objects combined)
    combined_mask_all = combine_masks(masks_to_combine)

    filename = f'combined_{iteration}.png'
    combined_image_path = os.path.join(images_output_folder, filename)
    segmentation_mask_path = os.path.join(leaf_instances_output_folder, filename)
    combined_mask_path = os.path.join(plant_instances_output_folder, filename)
    semantic_mask_path = os.path.join(semantics_output_folder, filename)

    cv2.imwrite(combined_image_path, combined_image)
    cv2.imwrite(segmentation_mask_path, segmentation_mask)
    cv2.imwrite(combined_mask_path, segmentation_mask if plant_is_leaf else combined_mask_all)
    cv2.imwrite(semantic_mask_path, combined_mask_all * SEMANTIC_VALUE)
    if VERBOSE:
        print(f"Iteration {iteration + 1}: Combined image saved to {combined_image_path}")
        print(f"Iteration {iteration + 1}: Segmentation mask saved to {segmentation_mask_path}")
        print(f"Iteration {iteration + 1}: Combined mask saved to {combined_mask_path}")
        print(f"Iteration {iteration + 1}: Semantics saved to {semantic_mask_path}")

if __name__ == "__main__":
    # Parameters
    folder_path = '.'  # Folder containing 'images', 'leaf_instances', and 'backgrounds' subfolders
    output_folder = '../urban_street_combined'  # Folder to save the results
    image_range = (9, 25)  # Range of number of images to combine per iteration (min, max)
    iterations = 10_000  # Number of iterations
    output_size = (512, 512)  # Size to which all images and masks will be resized (width, height)
    n = 8  # Number of subpixels per dimension for jittering

    process_images(folder_path, output_folder, image_range, iterations, output_size, n)
