import cv2
import os
import numpy as np

PATH = "data/finetune"

def main():
    for file in os.listdir(PATH + "/images"):
        img = cv2.imread(PATH + "/masks/" + file)
        cv2.imwrite(PATH + "/plant_instances/" + file, np.where(img != 0, np.ones(img.shape), np.zeros(img.shape)))
        cv2.imwrite(PATH + "/leaf_instances/" + file, np.where(img != 0, np.ones(img.shape), np.zeros(img.shape)))

if __name__ == '__main__':
    main()