import cv2
import os
import numpy as np

PATH = "data/finetune"

def main():
    for file in os.listdir(PATH + "/masks"):
        img = cv2.imread(PATH + "/masks/" + file)
        cv2.imwrite(PATH + "/semantics/" + file, np.where(img != 0, np.ones(img.shape) * 3, np.zeros(img.shape)))



if __name__ == '__main__':
    main()