import numpy as np
import cv2
import os
import tqdm

INPUT_PATH = "labels_color"
OUTPUT_PATH = "leaf_instances"

def main():
    pbar = tqdm(os.listdir(INPUT_PATH))
    for file in pbar:
        pbar.set_description("Processing: "+ file)
        img = cv2.imread(os.path.join(INPUT_PATH, file))
        file = file.replace("_label", "")
        pixels = img.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)
        unique_colors = np.array([color for color in unique_colors if not np.array_equal(color, [0, 0, 0])])
        leaf_instances = img.copy()

        for i, color in enumerate(unique_colors):
            mask = cv2.inRange(img, color, color)

            leaf_instances[mask == 255] = i
        leaf_instances = cv2.cvtColor(leaf_instances, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(OUTPUT_PATH, file), leaf_instances)

        plant_instance = cv2.inRange(img, (1, 1, 1), (255, 255, 255))
        plant_instance[plant_instance > 0] = 1

        cv2.imwrite(os.path.join("semantics", file), plant_instance)
        cv2.imwrite(os.path.join("plant_instances", file), plant_instance)
        cv2.imwrite(os.path.join("masks", file), plant_instance * 255)


if __name__ == '__main__':
    main()