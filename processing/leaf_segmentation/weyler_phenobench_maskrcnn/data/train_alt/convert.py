import cv2
import os

INPUT_DIR = "plant_instances"
OUTPUT_DIR = "masks"


def main():
    for file in os.listdir(INPUT_DIR):
        img = cv2.imread(os.path.join(INPUT_DIR, file))
        img[img > 0] = 255
        cv2.imwrite(os.path.join(OUTPUT_DIR, file), img)

if __name__ == "__main__":
    main()