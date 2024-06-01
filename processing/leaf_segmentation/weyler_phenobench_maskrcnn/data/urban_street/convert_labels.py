import numpy as np
import cv2
import os
from threading import Thread, current_thread
from tqdm import tqdm

INPUT_PATH = "labels"
OUTPUT_PATH = "leaf_instances"
THREADS = 1
SCALE_FACTOR = 4

def main(thread_num = 0, stop = None):
    pbar = tqdm(enumerate(os.listdir(INPUT_PATH)))
    for j, file in pbar:
        if stop is not None:
            if stop():
                break
        if j % THREADS != thread_num:
            continue
        pbar.set_description("Processing: " + file)

        rgb_img = cv2.imread(os.path.join("images_orig", file))
        rgb_img = cv2.resize(rgb_img, (rgb_img.shape[0] // SCALE_FACTOR, rgb_img.shape[1] // SCALE_FACTOR))
        cv2.imwrite(os.path.join("images", file), rgb_img)

        img = cv2.imread(os.path.join(INPUT_PATH, file))
        img = cv2.resize(img, (img.shape[0] // SCALE_FACTOR, img.shape[1] // SCALE_FACTOR))
        pixels = img.reshape(-1, 3)
        unique_colors = np.unique(pixels, axis=0)
        # remove black from unique colors
        unique_colors = np.array([color for color in unique_colors if not np.array_equal(color, [0, 0, 0])])
        leaf_instances = img.copy()

        for i, color in enumerate(unique_colors):
            mask = cv2.inRange(img, color, color)

            leaf_instances[mask == 255] = i
        leaf_instances = cv2.cvtColor(leaf_instances, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(OUTPUT_PATH, file), leaf_instances)

        #plant_instance = cv2.inRange(img, (1, 1, 1), (255, 255, 255))
        plant_instance_mask = np.any(img != [0, 0, 0], axis=-1)
        plant_instance = np.zeros(img.shape)
        #plant_instance[plant_instance_mask] = [1, 1, 1]
        plant_instance[plant_instance_mask] = [1, 1, 1]
        cv2.imshow("Instance", plant_instance * 255)
        cv2.waitKey(0)

        cv2.imwrite(os.path.join("semantics", file), plant_instance)
        cv2.imwrite(os.path.join("plant_instances", file), plant_instance)
        cv2.imwrite(os.path.join("masks", file), plant_instance * 255)


if __name__ == '__main__':
    main()
    exit()
    threads = list()
    stop_threads = False
    try:
        for i in range(0, THREADS):
            thread = Thread(target=main, args = (i, lambda : stop_threads, ))
            thread.start()
            threads.append(thread)
    except KeyboardInterrupt:
        for t in threads:
            stop_threads = True
            t.join()