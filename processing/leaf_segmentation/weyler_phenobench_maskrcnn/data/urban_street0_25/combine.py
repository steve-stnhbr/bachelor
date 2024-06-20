import cv2
import numpy as np
import os
import random

def apply_mask(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)

def transform_image(image, angle, scale, tx, ty):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    # Translation matrix
    rotation_matrix[0, 2] += tx
    rotation_matrix[1, 2] += ty

    # Apply the transformation
    transformed_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return transformed_image

def create_segmentation_mask(image, mask, value):
    segmented_mask = np.zeros_like(image)
    segmented_mask[mask > 0] = value
    return segmented_mask

def process_images(folder_path, output_folder, image_range, iterations):
    image_folder = os.path.join(folder_path, 'images')
    mask_folder = os.path.join(folder_path, 'masks')

    # Get the list of image and mask files
    image_files = sorted([f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))])
    mask_files = sorted([f for f in os.listdir(mask_folder) if os.path.isfile(os.path.join(mask_folder, f))])

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "masks"), exist_ok=True)

    for iteration in range(iterations):
        # Randomly pick the number of images to combine from the provided range
        num_images = random.randint(image_range[0], image_range[1])

        # Randomly select `num_images` from the folder
        selected_indices = random.sample(range(len(image_files)), num_images)

        # Initialize the combined image and segmentation mask
        first_image = cv2.imread(os.path.join(image_folder, image_files[selected_indices[0]]))
        combined_image = np.zeros_like(first_image)
        segmentation_mask = np.zeros_like(first_image)

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

            # Apply the mask to the image
            masked_image = apply_mask(image, mask)

            # Generate random transformations
            angle = random.uniform(-45, 45)  # Random rotation between -45 and 45 degrees
            scale = random.uniform(0.2, .7)  # Random scaling between 0.5 and 1.5
            tx = random.randint(-100, 100)  # Random translation in x direction
            ty = random.randint(-100, 100)  # Random translation in y direction

            # Apply transformations
            transformed_image = transform_image(masked_image, angle, scale, tx, ty)

            # Combine the transformed image into the combined image
            combined_image = cv2.add(combined_image, transformed_image)

            # Create the segmentation mask
            segmentation_value = [idx + 1, idx + 1, idx + 1]
            segmented_mask = create_segmentation_mask(masked_image, mask, segmentation_value)
            segmentation_mask = cv2.add(segmentation_mask, segmented_mask)

        # Save the results
        combined_image_path = os.path.join(output_folder, "images", f'combined_image_{iteration + 1}.jpg')
        segmentation_mask_path = os.path.join(output_folder,"masks", f'segmentation_mask_{iteration + 1}.png')
        cv2.imwrite(combined_image_path, combined_image)
        cv2.imwrite(segmentation_mask_path, segmentation_mask)

        print(f"Iteration {iteration + 1}: Combined image saved to {combined_image_path}")
        print(f"Iteration {iteration + 1}: Segmentation mask saved to {segmentation_mask_path}")

def main():
    # Parameters
    folder_path = '.'  # Folder containing 'images' and 'masks' subfolders
    output_folder = '../urban_street_combined'  # Folder to save the results
    image_range = (4, 10)  # Range of number of images to combine per iteration (min, max)
    iterations = 100  # Number of iterations

    process_images(folder_path, output_folder, image_range, iterations)

if __name__ == "__main__":
    main()
